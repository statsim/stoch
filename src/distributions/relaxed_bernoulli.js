import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertOneOf, assertPositive } from '../internal/assert-util'

/**
 * RelaxedBernoulli distribution (BinConcrete / Gumbel-Softmax for binary case).
 *
 * A continuous relaxation of the Bernoulli distribution parameterized by
 * temperature and probs/logits. Samples are in (0, 1) and approach
 * discrete {0, 1} as temperature → 0.
 *
 * The density on y ∈ (0,1) is the logistic-sigmoid of a Logistic distribution:
 *   z ~ Logistic(logits/temperature, 1/temperature)
 *   y = sigmoid(z)
 *
 * log p(y) = log(temperature) - temperature * logits + (temperature - 1) * log(y)
 *            + (temperature - 1) * log(1 - y) - 2 * log(y^temperature * exp(-logits)
 *            + (1-y)^temperature)
 *
 * Simplified: based on the Binary Concrete distribution (Maddison et al. 2016).
 */
export class RelaxedBernoulli extends Distribution {
  constructor({ temperature, probs, logits, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'RelaxedBernoulli'
    })

    assertOneOf(probs, logits, 'probs', 'logits')

    this._temperature = this._addParameter('temperature', temperature)

    if (this._validateArgs) {
      assertPositive(this._temperature, 'temperature')
    }

    if (probs != null) {
      this._probs = this._addParameter('probs', probs, 'float32')
      this._logits = null
    } else {
      this._logits = this._addParameter('logits', logits, 'float32')
      this._probs = null
    }
  }

  get temperature() { return this._temperature }

  get probs() {
    if (this._probs) return this._probs
    return tf.tidy(() => tf.sigmoid(this._logits))
  }

  get logits() {
    if (this._logits) return this._logits
    return tf.tidy(() => tf.log(tf.div(this._probs, tf.sub(1, this._probs))))
  }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]
    const u = tf.randomUniform(shape, 1e-7, 1 - 1e-7)
    // Logistic sample: logit(u) = log(u/(1-u))
    const logitU = tf.log(tf.div(u, tf.sub(1, u)))
    const logits = this._logits || tf.log(tf.div(this._probs, tf.sub(1, this._probs)))
    // z = (logits + logitU) / temperature
    const z = tf.div(tf.add(logits, logitU), this._temperature)
    return tf.sigmoid(z)
  }

  _logProb(value) {
    // Binary Concrete log density:
    // log p(y) = log(t) + (t-1)*[log(y) + log(1-y)] + logits
    //            - 2*log(y^t * exp(logits/t ?) + (1-y)^t)
    // More carefully: y = sigmoid(z/t + logits/t) where z ~ Logistic(0,1)
    // The density in terms of y:
    // log p(y) = log(t) - t*logit(y) + logits - 2*softplus(-t*logit(y) + logits)
    //          ... Let me use the standard BinConcrete formula:
    // log p(y) = log(t) + logits - (1+t)*log(y) - (1+t)*log(1-y)
    //            - 2*log(exp(logits/t)*y^(-1) + (1-y)^(-1))
    // Actually the clearest form:
    // Let L = logits, T = temperature
    // log p(y) = log(T) + L - T*log(y) - T*log(1-y) - 2*log(y^(-T)*exp(L) + (1-y)^(-T))
    // ... this is getting messy. Use the Logistic density + change of variables.
    // z = logit(y) * T - L, y = sigmoid((z + L)/T)
    // dy/dz = y*(1-y)/T
    // log p(y) = log p_logistic(z) - log(dy/dz) = log p_logistic(z) - log(y) - log(1-y) + log(T)
    // where p_logistic(z) is the standard Logistic density:
    // log p_logistic(z) = -z - 2*log(1 + exp(-z)) = -z - 2*softplus(-z)
    // But z here is the scaled logistic: z = logit(y)*T - L
    // The Logistic(L, T) density of w = logit(y): (already incorporating the sigmoid transform)
    // log p(w | L, T) = (w - L)/T - log(T) - 2*softplus((w - L)/T) ... wait no.
    // Logistic(mu, s) has density: exp(-(x-mu)/s) / (s * (1 + exp(-(x-mu)/s))²)
    // log density = -(x-mu)/s - log(s) - 2*log(1 + exp(-(x-mu)/s))
    //             = -(x-mu)/s - log(s) - 2*softplus(-(x-mu)/s)
    //
    // w = logit(y), y = sigmoid(w)
    // w ~ Logistic(L, T) ... no, we have y = sigmoid((L + logistic_noise)/T)
    // So w' = (L + logistic_noise)/T, and y = sigmoid(w')
    // logistic_noise ~ Logistic(0, 1)
    // w' ~ Logistic(L/T, 1/T)? No: if U ~ Logistic(0,1), then L + U ~ Logistic(L, 1),
    // then (L + U)/T ~ Logistic(L/T, 1/T).
    // So w' ~ Logistic(L/T, 1/T).
    //
    // y = sigmoid(w'), w' = logit(y)
    // p(y) = p_logistic(w') * |dw'/dy| = p_logistic(w') / (y*(1-y))
    //
    // log p(y) = log p_Logistic(L/T, 1/T)(logit(y)) - log(y) - log(1-y)
    //
    // log p_Logistic(mu, s)(x) = -(x-mu)/s - log(s) - 2*softplus(-(x-mu)/s)
    // mu = L/T, s = 1/T, x = logit(y)
    // (x - mu)/s = (logit(y) - L/T) * T = T*logit(y) - L
    //
    // log p(y) = -(T*logit(y) - L) - log(1/T) - 2*softplus(-(T*logit(y) - L))
    //            - log(y) - log(1-y)
    //          = L - T*logit(y) + log(T) - 2*softplus(L - T*logit(y))
    //            - log(y) - log(1-y)

    const logits = this._logits || tf.log(tf.div(this._probs, tf.sub(1, this._probs)))
    const T = this._temperature

    // logit(y) = log(y/(1-y))
    const logitY = tf.sub(tf.log(value), tf.log(tf.sub(1, value)))
    const tLogitY = tf.mul(T, logitY)
    const diff = tf.sub(logits, tLogitY)

    return tf.sub(
      tf.sub(tf.add(diff, tf.log(T)), tf.mul(2, tf.softplus(diff))),
      tf.add(tf.log(value), tf.log(tf.sub(1, value)))
    )
  }

  _mean() {
    return this._probs || tf.sigmoid(this._logits)
  }
}
