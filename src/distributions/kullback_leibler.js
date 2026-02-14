import * as tf from '@tensorflow/tfjs'

/**
 * KL divergence registry.
 *
 * Mirrors TFP Python's kullback_leibler.py.
 * Distributions register their KL divergence implementations,
 * and users call klDivergence(p, q) to compute KL(p || q).
 */

const _KL_REGISTRY = new Map()

/**
 * Register a KL divergence function for a pair of distribution types.
 *
 * @param {Function} typeP - constructor of distribution P
 * @param {Function} typeQ - constructor of distribution Q
 * @param {Function} fn - function(p, q) => tf.Tensor
 */
export function registerKL(typeP, typeQ, fn) {
  const key = `${typeP.name}_${typeQ.name}`
  _KL_REGISTRY.set(key, fn)
}

/**
 * Compute KL(p || q) = E_p[log(p/q)].
 *
 * @param {Distribution} p
 * @param {Distribution} q
 * @returns {tf.Tensor}
 */
export function klDivergence(p, q) {
  const key = `${p.constructor.name}_${q.constructor.name}`
  const fn = _KL_REGISTRY.get(key)
  if (!fn) {
    throw new Error(`No KL divergence registered for ${p.constructor.name} and ${q.constructor.name}`)
  }
  return tf.tidy(() => fn(p, q))
}

// --- Built-in KL implementations ---

// KL(Normal || Normal)
// = log(σ_q / σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5
import { Normal } from './normal'
registerKL(Normal, Normal, (p, q) => {
  const logRatio = tf.log(tf.div(q.scale, p.scale))
  const varRatio = tf.div(tf.square(p.scale), tf.square(q.scale))
  const meanDiff = tf.div(tf.square(tf.sub(p.loc, q.loc)), tf.square(q.scale))
  return tf.add(logRatio, tf.sub(tf.mul(0.5, tf.add(varRatio, meanDiff)), 0.5))
})

// KL(Bernoulli || Bernoulli)
// = p*log(p/q) + (1-p)*log((1-p)/(1-q))
import { Bernoulli } from './bernoulli'
registerKL(Bernoulli, Bernoulli, (p, q) => {
  const pp = p.probs
  const qp = q.probs
  const t1 = tf.mul(pp, tf.log(tf.div(pp, qp)))
  const t2 = tf.mul(tf.sub(1, pp), tf.log(tf.div(tf.sub(1, pp), tf.sub(1, qp))))
  return tf.add(t1, t2)
})

// KL(Gamma || Gamma)
// = (a_p - a_q) * ψ(a_p) - logΓ(a_p) + logΓ(a_q)
//   + a_q * log(b_p / b_q) + a_p * (b_q / b_p - 1)
// where a = concentration, b = rate
import { Gamma } from './gamma'
import { logGamma, digamma } from '../math/special'
registerKL(Gamma, Gamma, (p, q) => {
  const ap = p.concentration
  const aq = q.concentration
  const bp = p.rate
  const bq = q.rate
  const psiAp = digamma(ap)
  return tf.add(
    tf.add(
      tf.sub(tf.mul(tf.sub(ap, aq), psiAp), logGamma(ap)),
      logGamma(aq)
    ),
    tf.add(
      tf.mul(aq, tf.log(tf.div(bp, bq))),
      tf.mul(ap, tf.sub(tf.div(bq, bp), 1))
    )
  )
})

// KL(Beta || Beta)
// = logB(a_q, b_q) - logB(a_p, b_p) + (a_p - a_q)*ψ(a_p) + (b_p - b_q)*ψ(b_p)
//   + (a_q - a_p + b_q - b_p) * ψ(a_p + b_p)
import { Beta } from './beta'
import { logBeta } from '../math/special'
registerKL(Beta, Beta, (p, q) => {
  const ap = p.concentration1
  const bp = p.concentration0
  const aq = q.concentration1
  const bq = q.concentration0
  const psiAp = digamma(ap)
  const psiBp = digamma(bp)
  const psiApBp = digamma(tf.add(ap, bp))
  return tf.add(
    tf.add(
      tf.sub(logBeta(aq, bq), logBeta(ap, bp)),
      tf.mul(tf.sub(ap, aq), psiAp)
    ),
    tf.add(
      tf.mul(tf.sub(bp, bq), psiBp),
      tf.mul(tf.add(tf.sub(aq, ap), tf.sub(bq, bp)), psiApBp)
    )
  )
})

// KL(Exponential || Exponential)
// = log(λ_q / λ_p) + λ_p / λ_q - 1
import { Exponential } from './exponential'
registerKL(Exponential, Exponential, (p, q) => {
  const rp = p.rate
  const rq = q.rate
  return tf.sub(tf.add(tf.log(tf.div(rq, rp)), tf.div(rp, rq)), 1)
})

// KL(Dirichlet || Dirichlet)
// = logB(α_q) - logB(α_p) + sum_i (α_p_i - α_q_i) * (ψ(α_p_i) - ψ(Σα_p))
import { Dirichlet } from './dirichlet'
registerKL(Dirichlet, Dirichlet, (p, q) => {
  const ap = p.concentration
  const aq = q.concentration
  const sumAp = tf.sum(ap, -1)
  const sumAq = tf.sum(aq, -1)
  // logB(α) = sum(logΓ(α_i)) - logΓ(sum(α_i))
  const logBp = tf.sub(tf.sum(logGamma(ap), -1), logGamma(sumAp))
  const logBq = tf.sub(tf.sum(logGamma(aq), -1), logGamma(sumAq))
  const psiAp = digamma(ap)
  const psiSumAp = digamma(sumAp)
  return tf.add(
    tf.sub(logBq, logBp),
    tf.sum(tf.mul(tf.sub(ap, aq), tf.sub(psiAp, psiSumAp)), -1)
  )
})

// KL(Categorical || Categorical)
// = sum_i p_i * log(p_i / q_i)
import { Categorical } from './categorical'
registerKL(Categorical, Categorical, (p, q) => {
  const pp = p.probs
  const qp = q.probs
  const logRatio = tf.sub(tf.log(pp), tf.log(qp))
  return tf.sum(tf.mul(pp, logRatio), -1)
})

// KL(Laplace || Laplace)
// = |μ_p - μ_q| / b_q + log(b_q / b_p) + b_p / b_q * exp(-|μ_p - μ_q| / b_p) - 1
// Simplified: = log(b_q/b_p) + |μ_p - μ_q|/b_q + b_p*exp(-|Δμ|/b_p)/b_q - 1
// Actually, exact formula:
// KL = log(b_q/b_p) + |μ_p - μ_q|/b_q + (b_p/b_q)*exp(-|μ_p - μ_q|/b_p) - 1
import { Laplace } from './laplace'
registerKL(Laplace, Laplace, (p, q) => {
  const absDiff = tf.abs(tf.sub(p.loc, q.loc))
  const bp = p.scale
  const bq = q.scale
  return tf.sub(
    tf.add(
      tf.add(
        tf.log(tf.div(bq, bp)),
        tf.div(absDiff, bq)
      ),
      tf.mul(tf.div(bp, bq), tf.exp(tf.neg(tf.div(absDiff, bp))))
    ),
    1
  )
})
