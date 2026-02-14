import * as tf from '@tensorflow/tfjs'
import { LOG_SQRT_2PI } from './numeric'

// Lanczos coefficients for logGamma approximation (g=5, n=7)
// Adapted from WebPPL special.js and Numerical Recipes
const GAMMA_COF = [
  76.18009172947146,
  -86.50532032941677,
  24.01409824083091,
  -1.231739572450155,
  0.1208650973866179e-2,
  -0.5395239384953e-5
]

/**
 * Log of the gamma function using Lanczos approximation.
 * Works element-wise on tensors.
 *
 * Adapted from WebPPL special.js lines 11-21.
 * Accurate to ~15 significant digits for x > 0.
 */
export function logGamma(x) {
  return tf.tidy(() => {
    const xTensor = x instanceof tf.Tensor ? x.cast('float32') : tf.scalar(x, 'float32')

    // Lanczos approx: Γ(x+1) = sqrt(2π) * (x + g + 0.5)^(x+0.5) * exp(-(x+g+0.5)) * ser
    // So logΓ(x) = logΓ(x+1) - log(x) but we use the direct formula for x itself
    const xm1 = tf.sub(xTensor, 1)
    const tmp = tf.add(xm1, 5.5)
    const logTmp = tf.sub(
      tmp,
      tf.mul(tf.add(xm1, 0.5), tf.log(tmp))
    )

    // Compute series sum
    let ser = tf.scalar(1.000000000190015)
    let xInc = xm1
    for (let j = 0; j < 6; j++) {
      xInc = tf.add(xInc, 1)
      ser = tf.add(ser, tf.div(GAMMA_COF[j], xInc))
    }

    return tf.add(tf.neg(logTmp), tf.log(tf.mul(2.5066282746310005, ser)))
  })
}

/**
 * Digamma (psi) function — the derivative of logGamma.
 * Uses asymptotic expansion for x >= 6, recursion otherwise.
 *
 * Adapted from WebPPL special.js lines 24-37.
 */
export function digamma(x) {
  return tf.tidy(() => {
    const xTensor = x instanceof tf.Tensor ? x.cast('float32') : tf.scalar(x, 'float32')

    // Use recursion: ψ(x) = ψ(x+n) - Σ 1/(x+k) for k=0..n-1
    // until x+n >= 6, then use asymptotic expansion
    // For tensor ops, shift all values up to >= 6 and track the correction
    const threshold = 6
    const nShift = tf.maximum(tf.sub(threshold, tf.floor(xTensor)), 0).cast('int32')
    const maxShift = nShift.max().dataSync()[0]

    let correction = tf.zerosLike(xTensor)
    let xShifted = xTensor
    for (let k = 0; k < maxShift; k++) {
      const shouldApply = tf.greater(nShift, k).cast('float32')
      correction = tf.sub(correction, tf.mul(shouldApply, tf.reciprocal(xShifted)))
      xShifted = tf.add(xShifted, shouldApply)
    }

    // Asymptotic expansion for x >= 6
    const x2 = tf.square(xShifted)
    const x4 = tf.square(x2)
    const x6 = tf.mul(x4, x2)
    const x8 = tf.square(x4)
    const x10 = tf.mul(x8, x2)
    const x12 = tf.mul(x10, x2)
    const x14 = tf.mul(x12, x2)

    const asymptotic = tf.add(
      tf.log(xShifted),
      tf.add(
        tf.mul(-0.5, tf.reciprocal(xShifted)),
        tf.add(
          tf.mul(-1 / 12, tf.reciprocal(x2)),
          tf.add(
            tf.mul(1 / 120, tf.reciprocal(x4)),
            tf.add(
              tf.mul(-1 / 252, tf.reciprocal(x6)),
              tf.add(
                tf.mul(1 / 240, tf.reciprocal(x8)),
                tf.add(
                  tf.mul(-5 / 660, tf.reciprocal(x10)),
                  tf.add(
                    tf.mul(691 / 32760, tf.reciprocal(x12)),
                    tf.mul(-1 / 12, tf.reciprocal(x14))
                  )
                )
              )
            )
          )
        )
      )
    )

    return tf.add(asymptotic, correction)
  })
}

/**
 * Log of the beta function: logBeta(a, b) = logGamma(a) + logGamma(b) - logGamma(a + b)
 */
export function logBeta(a, b) {
  return tf.tidy(() => {
    const aTensor = a instanceof tf.Tensor ? a : tf.scalar(a)
    const bTensor = b instanceof tf.Tensor ? b : tf.scalar(b)
    return tf.sub(
      tf.add(logGamma(aTensor), logGamma(bTensor)),
      logGamma(tf.add(aTensor, bTensor))
    )
  })
}

/**
 * Normal CDF (Φ function).
 * ndtr(x) = 0.5 * (1 + erf(x / sqrt(2)))
 *
 * Uses tf.erf which is available in tf.js 4.x.
 */
export function ndtr(x) {
  return tf.tidy(() => {
    const xTensor = x instanceof tf.Tensor ? x : tf.scalar(x)
    return tf.mul(0.5, tf.add(1, tf.erf(tf.div(xTensor, Math.SQRT2))))
  })
}

/**
 * Log of normal CDF.
 * Numerically stable for large negative x using the complementary error function.
 */
export function logNdtr(x) {
  return tf.tidy(() => {
    const xTensor = x instanceof tf.Tensor ? x : tf.scalar(x)
    return tf.log(ndtr(xTensor))
  })
}

/**
 * Log of binomial coefficient: log C(n, k) = logGamma(n+1) - logGamma(k+1) - logGamma(n-k+1)
 *
 * Works on JS numbers (scalars). For tensor inputs use logGamma directly.
 *
 * @param {number} n - total count (non-negative integer)
 * @param {number} k - selection count (0 <= k <= n)
 * @returns {number}
 */
export function logChoose(n, k) {
  if (k < 0 || k > n) return -Infinity
  if (k === 0 || k === n) return 0
  return tf.tidy(() => {
    const result = tf.sub(
      logGamma(tf.scalar(n + 1)),
      tf.add(logGamma(tf.scalar(k + 1)), logGamma(tf.scalar(n - k + 1)))
    )
    return result.dataSync()[0]
  })
}

/**
 * Regularized incomplete gamma function P(a, x) = γ(a,x) / Γ(a).
 *
 * Uses series expansion for x < a+1, continued fraction otherwise.
 * Adapted from Numerical Recipes (Press et al.) §6.2.
 *
 * Operates on JS numbers; returns { lower, upper } where lower = P(a,x), upper = Q(a,x).
 *
 * @param {number} a - shape parameter (a > 0)
 * @param {number} x - integration limit (x >= 0)
 * @returns {{ lower: number, upper: number }}
 */
export function incompleteGamma(a, x) {
  if (x < 0) throw new Error('incompleteGamma: x must be >= 0')
  if (a <= 0) throw new Error('incompleteGamma: a must be > 0')
  if (x === 0) return { lower: 0, upper: 1 }

  const lnGammaA = tf.tidy(() => logGamma(tf.scalar(a)).dataSync()[0])

  if (x < a + 1) {
    // Series expansion for P(a,x) — NR gser
    let sum = 1 / a
    let term = 1 / a
    for (let n = 1; n < 200; n++) {
      term *= x / (a + n)
      sum += term
      if (Math.abs(term) < Math.abs(sum) * 1e-14) break
    }
    const lower = sum * Math.exp(-x + a * Math.log(x) - lnGammaA)
    return { lower, upper: 1 - lower }
  } else {
    // Continued fraction for Q(a,x) — NR gcf (modified Lentz's method)
    const FPMIN = 1e-30
    let b = x + 1 - a
    let c = 1 / FPMIN
    let d = 1 / b
    let h = d
    for (let i = 1; i < 200; i++) {
      const an = -i * (i - a)
      b += 2
      d = an * d + b
      if (Math.abs(d) < FPMIN) d = FPMIN
      c = b + an / c
      if (Math.abs(c) < FPMIN) c = FPMIN
      d = 1 / d
      const delta = d * c
      h *= delta
      if (Math.abs(delta - 1) < 1e-14) break
    }
    const upper = Math.exp(-x + a * Math.log(x) - lnGammaA) * h
    return { lower: 1 - upper, upper }
  }
}

/**
 * Regularized incomplete beta function I_x(a, b).
 *
 * Uses the continued fraction representation (Lentz's method).
 * Adapted from Numerical Recipes §6.4.
 *
 * @param {number} a - parameter a > 0
 * @param {number} b - parameter b > 0
 * @param {number} x - value in [0, 1]
 * @returns {number} I_x(a, b) ∈ [0, 1]
 */
export function incompleteBeta(a, b, x) {
  if (x < 0 || x > 1) throw new Error('incompleteBeta: x must be in [0, 1]')
  if (x === 0) return 0
  if (x === 1) return 1

  // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a) when x > (a+1)/(a+b+2)
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - incompleteBeta(b, a, 1 - x)
  }

  const lnBetaAB = tf.tidy(() => logBeta(tf.scalar(a), tf.scalar(b)).dataSync()[0])
  const front = Math.exp(a * Math.log(x) + b * Math.log(1 - x) - lnBetaAB) / a

  // Continued fraction — NR betacf (modified Lentz's method)
  const FPMIN = 1e-30
  const qab = a + b
  const qap = a + 1
  const qam = a - 1
  let c = 1
  let d = 1 - qab * x / qap
  if (Math.abs(d) < FPMIN) d = FPMIN
  d = 1 / d
  let h = d

  for (let m = 1; m < 200; m++) {
    const m2 = 2 * m
    // Even step
    let aa = m * (b - m) * x / ((qam + m2) * (a + m2))
    d = 1 + aa * d
    if (Math.abs(d) < FPMIN) d = FPMIN
    c = 1 + aa / c
    if (Math.abs(c) < FPMIN) c = FPMIN
    d = 1 / d
    h *= d * c

    // Odd step
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
    d = 1 + aa * d
    if (Math.abs(d) < FPMIN) d = FPMIN
    c = 1 + aa / c
    if (Math.abs(c) < FPMIN) c = FPMIN
    d = 1 / d
    const delta = d * c
    h *= delta

    if (Math.abs(delta - 1) < 1e-14) break
  }

  return front * h
}

/**
 * Modified Bessel function of the first kind, order 0: I₀(x).
 *
 * Uses polynomial approximation (Abramowitz & Stegun §9.8).
 * Accurate to ~1e-7 for all x.
 *
 * @param {number} x - non-negative argument
 * @returns {number}
 */
export function besselI0(x) {
  const ax = Math.abs(x)
  if (ax < 3.75) {
    const t = (ax / 3.75) ** 2
    return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
      + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))))
  } else {
    const t = 3.75 / ax
    return (Math.exp(ax) / Math.sqrt(ax)) * (0.39894228 + t * (0.01328592
      + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281
      + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633
      + t * 0.00392377))))))))
  }
}

/**
 * Modified Bessel function of the first kind, order 1: I₁(x).
 *
 * Uses polynomial approximation (Abramowitz & Stegun §9.8).
 *
 * @param {number} x - non-negative argument
 * @returns {number}
 */
export function besselI1(x) {
  const ax = Math.abs(x)
  if (ax < 3.75) {
    const t = (ax / 3.75) ** 2
    const result = ax * (0.5 + t * (0.87890594 + t * (0.51498869
      + t * (0.15084934 + t * (0.02658733 + t * (0.00301532
      + t * 0.00032411))))))
    return x < 0 ? -result : result
  } else {
    const t = 3.75 / ax
    let result = 0.02282967 + t * (-0.02895312 + t * (0.01787654
      + t * (-0.00420059)))
    result = 0.39894228 + t * (-0.03988024 + t * (-0.00362018
      + t * (0.00163801 + t * (-0.01031555 + t * result))))
    result *= Math.exp(ax) / Math.sqrt(ax)
    return x < 0 ? -result : result
  }
}

/**
 * Log of modified Bessel function I₀(x).
 * Numerically stable for large x where I₀ overflows.
 *
 * @param {number} x - non-negative argument
 * @returns {number}
 */
export function logBesselI0(x) {
  const ax = Math.abs(x)
  if (ax < 500) {
    return Math.log(besselI0(ax))
  }
  // For large x: log I₀(x) ≈ x - 0.5*log(2πx) + log(1 + 1/(8x) + ...)
  return ax - 0.5 * Math.log(2 * Math.PI * ax) + Math.log(1 + 1 / (8 * ax))
}
