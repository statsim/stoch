import * as tf from '@tensorflow/tfjs'

/**
 * Abstract base class for GP kernels (covariance functions).
 *
 * Subclasses must implement:
 *   _apply(x1, x2) - compute k(x1, x2) for individual pairs
 *   matrix(x1, x2) - compute the full kernel matrix K[i,j] = k(x1[i], x2[j])
 */
export class Kernel {
  constructor({ name = 'Kernel' } = {}) {
    this._name = name
  }

  get name() { return this._name }

  /**
   * Compute the kernel matrix K[i,j] = k(x1[i], x2[j]).
   *
   * @param {tf.Tensor} x1 - shape [n1, d]
   * @param {tf.Tensor} x2 - shape [n2, d]
   * @returns {tf.Tensor} shape [n1, n2]
   */
  matrix(x1, x2) {
    return tf.tidy(() => this._matrix(x1, x2))
  }

  _matrix(x1, x2) {
    throw new Error(`${this._name}._matrix not implemented`)
  }

  /**
   * Compute k(x1, x2) pointwise (for equal-length inputs).
   *
   * @param {tf.Tensor} x1 - shape [n, d]
   * @param {tf.Tensor} x2 - shape [n, d]
   * @returns {tf.Tensor} shape [n]
   */
  apply(x1, x2) {
    return tf.tidy(() => this._apply(x1, x2))
  }

  _apply(x1, x2) {
    throw new Error(`${this._name}._apply not implemented`)
  }
}
