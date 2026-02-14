import * as tf from '@tensorflow/tfjs'

/**
 * Numerically stable log(1 - exp(x)) for x <= 0.
 * Uses the identity: log(1 - exp(x)) = log1p(-exp(x)) for x < -log(2)
 *                                     = log(-expm1(x))  for x >= -log(2)
 */
export function log1mexp(x) {
  return tf.tidy(() => {
    const xTensor = x instanceof tf.Tensor ? x : tf.scalar(x)
    const threshold = -Math.LN2
    // For x < threshold: log1p(-exp(x)) is more stable
    // For x >= threshold: log(-expm1(x)) is more stable
    const useLog1p = xTensor.less(threshold)
    const branch1 = tf.log1p(tf.exp(xTensor).neg())
    const branch2 = tf.log(tf.sub(tf.scalar(1), tf.exp(xTensor)).abs().clipByValue(1e-38, Infinity))
    return tf.where(useLog1p, branch1, branch2)
  })
}

/**
 * Numerically stable log(exp(a) + exp(b)).
 * This is a convenience wrapper — tf.logSumExp handles the general case.
 */
export function logAddExp(a, b) {
  return tf.tidy(() => {
    const aTensor = a instanceof tf.Tensor ? a : tf.scalar(a)
    const bTensor = b instanceof tf.Tensor ? b : tf.scalar(b)
    const max = tf.maximum(aTensor, bTensor)
    return tf.add(max, tf.log(
      tf.add(tf.exp(tf.sub(aTensor, max)), tf.exp(tf.sub(bTensor, max)))
    ))
  })
}

/**
 * Inverse of softplus: log(exp(x) - 1).
 * Numerically stable for large x where it approaches x.
 */
export function softplusInverse(x) {
  return tf.tidy(() => {
    const xTensor = x instanceof tf.Tensor ? x : tf.scalar(x)
    // For large x, softplus_inverse(x) ≈ x
    // For small x, use log(expm1(x))
    const threshold = tf.scalar(20)
    const isLarge = xTensor.greater(threshold)
    const stableResult = tf.log(tf.sub(tf.exp(xTensor), tf.scalar(1)).clipByValue(1e-38, Infinity))
    return tf.where(isLarge, xTensor, stableResult)
  })
}
