import * as tf from '@tensorflow/tfjs'

/**
 * Inverse of the normal CDF (quantile function / probit function).
 * ndtri(p) returns x such that Φ(x) = p, where Φ is the standard normal CDF.
 *
 * Uses Acklam's algorithm with rational polynomial approximation.
 * Reference: Peter J. Acklam's "An algorithm for computing the inverse
 * normal cumulative distribution function" (2003).
 *
 * Accuracy: |relative error| < 1.15e-9 across full range.
 *
 * The algorithm divides [0,1] into three regions:
 *   - Central region (0.02425 <= p <= 0.97575): rational approximation
 *   - Lower tail (p < 0.02425): asymptotic expansion
 *   - Upper tail (p > 0.97575): symmetry with lower tail
 */

// Coefficients for rational approximation in the central region
const A = [
  -3.969683028665376e+01,
   2.209460984245205e+02,
  -2.759285104469687e+02,
   1.383577518672690e+02,
  -3.066479806614716e+01,
   2.506628277459239e+00
]

const B = [
  -5.447609879822406e+01,
   1.615858368580409e+02,
  -1.556989798598866e+02,
   6.680131188771972e+01,
  -1.328068155288572e+01
]

// Coefficients for rational approximation in the tail regions
const C = [
  -7.784894002430293e-03,
  -3.223964580411365e-01,
  -2.400758277161838e+00,
  -2.549732539343734e+00,
   4.374664141464968e+00,
   2.938163982698783e+00
]

const D = [
   7.784695709041462e-03,
   3.224671290700398e-01,
   2.445134137142996e+00,
   3.754408661907416e+00
]

const P_LOW = 0.02425
const P_HIGH = 1 - P_LOW

export function ndtri(p) {
  return tf.tidy(() => {
    const pTensor = p instanceof tf.Tensor ? p.cast('float32') : tf.scalar(p, 'float32')

    // Central region: 0.02425 <= p <= 0.97575
    const q = tf.sub(pTensor, 0.5)
    const r_central = tf.square(q)

    const num_central = tf.add(
      tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(
        tf.mul(r_central, A[0]), A[1]),
        r_central), A[2]),
        r_central), A[3]),
        r_central), A[4]),
        r_central), A[5])

    const den_central = tf.add(
      tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(
        tf.mul(r_central, B[0]), B[1]),
        r_central), B[2]),
        r_central), B[3]),
        r_central), B[4]),
        r_central), 1)

    const central_result = tf.div(tf.mul(q, num_central), den_central)

    // Lower tail: p < 0.02425
    const r_low = tf.sqrt(tf.neg(tf.mul(2, tf.log(pTensor.clipByValue(1e-38, 1)))))

    const num_low = tf.add(
      tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(
        tf.mul(r_low, C[0]), C[1]),
        r_low), C[2]),
        r_low), C[3]),
        r_low), C[4]),
        r_low), C[5])

    const den_low = tf.add(
      tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(
        tf.mul(r_low, D[0]), D[1]),
        r_low), D[2]),
        r_low), D[3]),
        r_low), 1)

    const low_result = tf.div(num_low, den_low)

    // Upper tail: use symmetry ndtri(p) = -ndtri(1-p)
    const r_high = tf.sqrt(tf.neg(tf.mul(2, tf.log(tf.sub(1, pTensor).clipByValue(1e-38, 1)))))

    const num_high = tf.add(
      tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(
        tf.mul(r_high, C[0]), C[1]),
        r_high), C[2]),
        r_high), C[3]),
        r_high), C[4]),
        r_high), C[5])

    const den_high = tf.add(
      tf.mul(tf.add(tf.mul(tf.add(tf.mul(tf.add(
        tf.mul(r_high, D[0]), D[1]),
        r_high), D[2]),
        r_high), D[3]),
        r_high), 1)

    const high_result = tf.neg(tf.div(num_high, den_high))

    // Select the appropriate result for each element
    const isLow = pTensor.less(P_LOW)
    const isHigh = pTensor.greater(P_HIGH)

    let result = central_result
    result = tf.where(isLow, low_result, result)
    result = tf.where(isHigh, high_result, result)

    return result
  })
}
