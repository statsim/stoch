import * as tf from '@tensorflow/tfjs'

/**
 * Assert that a value is a positive number or tensor of positive values.
 */
export function assertPositive(value, name) {
  if (value instanceof tf.Tensor) {
    const minTensor = value.min()
    const min = minTensor.dataSync()[0]
    minTensor.dispose()
    if (min <= 0) {
      throw new Error(`${name} must be positive, got min value ${min}`)
    }
  } else if (typeof value === 'number') {
    if (value <= 0 || isNaN(value)) {
      throw new Error(`${name} must be positive, got ${value}`)
    }
  }
}

/**
 * Assert that a value is non-negative.
 */
export function assertNonNegative(value, name) {
  if (value instanceof tf.Tensor) {
    const minTensor = value.min()
    const min = minTensor.dataSync()[0]
    minTensor.dispose()
    if (min < 0) {
      throw new Error(`${name} must be non-negative, got min value ${min}`)
    }
  } else if (typeof value === 'number') {
    if (value < 0 || isNaN(value)) {
      throw new Error(`${name} must be non-negative, got ${value}`)
    }
  }
}

/**
 * Assert that a value is in the range [low, high].
 */
export function assertInRange(value, low, high, name) {
  if (value instanceof tf.Tensor) {
    const [min, max] = tf.tidy(() => [value.min(), value.max()])
    const minVal = min.dataSync()[0]
    const maxVal = max.dataSync()[0]
    min.dispose()
    max.dispose()
    if (minVal < low || maxVal > high) {
      throw new Error(`${name} must be in [${low}, ${high}], got range [${minVal}, ${maxVal}]`)
    }
  } else if (typeof value === 'number') {
    if (value < low || value > high || isNaN(value)) {
      throw new Error(`${name} must be in [${low}, ${high}], got ${value}`)
    }
  }
}

/**
 * Assert that exactly one of two values is provided (not both, not neither).
 */
export function assertOneOf(a, b, nameA, nameB) {
  const hasA = a != null
  const hasB = b != null
  if (hasA === hasB) {
    throw new Error(`Exactly one of ${nameA} or ${nameB} must be provided`)
  }
}
