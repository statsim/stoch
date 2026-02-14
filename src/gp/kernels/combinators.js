import * as tf from '@tensorflow/tfjs'
import { Kernel } from './kernel'

/**
 * Sum of two kernels: k(x1, x2) = k1(x1, x2) + k2(x1, x2)
 */
export class Add extends Kernel {
  constructor(k1, k2) {
    super({ name: `Add(${k1.name}, ${k2.name})` })
    this._k1 = k1
    this._k2 = k2
  }

  _matrix(x1, x2) {
    return tf.add(this._k1.matrix(x1, x2), this._k2.matrix(x1, x2))
  }

  _apply(x1, x2) {
    return tf.add(this._k1.apply(x1, x2), this._k2.apply(x1, x2))
  }
}

/**
 * Product of two kernels: k(x1, x2) = k1(x1, x2) * k2(x1, x2)
 */
export class Product extends Kernel {
  constructor(k1, k2) {
    super({ name: `Product(${k1.name}, ${k2.name})` })
    this._k1 = k1
    this._k2 = k2
  }

  _matrix(x1, x2) {
    return tf.mul(this._k1.matrix(x1, x2), this._k2.matrix(x1, x2))
  }

  _apply(x1, x2) {
    return tf.mul(this._k1.apply(x1, x2), this._k2.apply(x1, x2))
  }
}

/**
 * Scaled kernel: k(x1, x2) = scale * k_inner(x1, x2)
 */
export class Scale extends Kernel {
  constructor(kernel, scale) {
    super({ name: `Scale(${kernel.name}, ${scale})` })
    this._kernel = kernel
    this._scale = scale
  }

  _matrix(x1, x2) {
    return tf.mul(this._scale, this._kernel.matrix(x1, x2))
  }

  _apply(x1, x2) {
    return tf.mul(this._scale, this._kernel.apply(x1, x2))
  }
}
