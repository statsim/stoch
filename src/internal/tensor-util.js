import * as tf from '@tensorflow/tfjs'

/**
 * Convert a value to a tf.Tensor if it isn't one already.
 * Accepts: number, array, tf.Tensor.
 */
export function toTensor(value, dtype) {
  if (value instanceof tf.Tensor) {
    return dtype ? value.cast(dtype) : value
  }
  return tf.tensor(value, undefined, dtype || 'float32')
}

/**
 * Compute the broadcast shape of two shapes.
 * Follows numpy broadcasting rules.
 *
 * @param {number[]} shapeA
 * @param {number[]} shapeB
 * @returns {number[]}
 */
export function broadcastShapes(shapeA, shapeB) {
  const maxLen = Math.max(shapeA.length, shapeB.length)
  const result = new Array(maxLen)

  for (let i = 0; i < maxLen; i++) {
    const a = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : 1
    const b = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1

    if (a === 1) {
      result[maxLen - 1 - i] = b
    } else if (b === 1) {
      result[maxLen - 1 - i] = a
    } else if (a === b) {
      result[maxLen - 1 - i] = a
    } else {
      throw new Error(
        `Shapes [${shapeA}] and [${shapeB}] are not broadcast-compatible`
      )
    }
  }

  return result
}

/**
 * Compute the broadcast shape of multiple shapes.
 *
 * @param {...number[]} shapes
 * @returns {number[]}
 */
export function broadcastShapesMultiple(...shapes) {
  if (shapes.length === 0) return []
  let result = shapes[0]
  for (let i = 1; i < shapes.length; i++) {
    result = broadcastShapes(result, shapes[i])
  }
  return result
}

/**
 * Convert a sample shape (user-provided) to a total number of samples.
 * E.g., [2, 3] → 6, [1000] → 1000, [] → 1
 */
export function shapeSize(shape) {
  if (!shape || shape.length === 0) return 1
  return shape.reduce((a, b) => a * b, 1)
}
