import * as tf from '@tensorflow/tfjs'

/**
 * Utilities for marshaling multi-parameter MCMC states.
 *
 * tf.grads() expects (Tensor[] → Tensor), but MCMC kernels work with
 * state objects like { mu: Tensor, sigma: Tensor }. This module bridges
 * between the two representations.
 */

/**
 * Convert a state object to an array of tensors + keys.
 * @param {Object|tf.Tensor} state
 * @returns {{ values: tf.Tensor[], keys: string[]|null }}
 */
export function stateToArray(state) {
  if (state instanceof tf.Tensor) {
    return { values: [state], keys: null }
  }
  const keys = Object.keys(state)
  const values = keys.map(k => state[k])
  return { values, keys }
}

/**
 * Convert an array of tensors back to a state object (or single tensor).
 * @param {tf.Tensor[]} values
 * @param {string[]|null} keys - null for single-tensor states
 * @returns {Object|tf.Tensor}
 */
export function arrayToState(values, keys) {
  if (keys === null) return values[0]
  const state = {}
  for (let i = 0; i < keys.length; i++) {
    state[keys[i]] = values[i]
  }
  return state
}

/**
 * Clone a state (tensor or object of tensors).
 * @param {Object|tf.Tensor} state
 * @returns {Object|tf.Tensor}
 */
export function cloneState(state) {
  if (state instanceof tf.Tensor) {
    return state.clone()
  }
  const cloned = {}
  for (const [k, v] of Object.entries(state)) {
    cloned[k] = v.clone()
  }
  return cloned
}

/**
 * Dispose a state (tensor or object of tensors).
 * @param {Object|tf.Tensor} state
 */
export function disposeState(state) {
  if (state instanceof tf.Tensor) {
    if (!state.isDisposed) state.dispose()
    return
  }
  for (const v of Object.values(state)) {
    if (v instanceof tf.Tensor && !v.isDisposed) v.dispose()
  }
}

/**
 * Wrap a target log-prob function that takes an object/tensor state
 * into one that takes individual tensor args (for use with tf.grads).
 * tf.grads(f) expects f(...tensors), returns gradF(tensorArray).
 * @param {Function} fn - (state) → scalar Tensor
 * @param {string[]|null} keys
 * @returns {Function} (...Tensor) → scalar Tensor
 */
export function wrapTargetLogProbFn(fn, keys) {
  return (...values) => fn(arrayToState(values, keys))
}

/**
 * Compute the value and gradients of targetLogProbFn at the given state.
 * Handles both single-tensor and object states.
 *
 * @param {Function} targetLogProbFn - (state) → scalar Tensor
 * @param {Object|tf.Tensor} state
 * @returns {{ value: tf.Tensor, grads: Object|tf.Tensor }}
 */
export function computeGrads(targetLogProbFn, state) {
  const { values, keys } = stateToArray(state)
  const wrappedFn = wrapTargetLogProbFn(targetLogProbFn, keys)
  const gradFn = tf.grads(wrappedFn)
  const gradsArray = gradFn(values)
  const value = tf.tidy(() => wrappedFn(...values))

  return {
    value,
    grads: arrayToState(gradsArray, keys)
  }
}

/**
 * Compute the value and gradients together using tf.variableGrads.
 * More efficient than separate forward + backward passes.
 *
 * @param {Function} targetLogProbFn - (state) → scalar Tensor
 * @param {Object|tf.Tensor} state
 * @returns {{ value: tf.Tensor, grads: Object|tf.Tensor }}
 */
export function valueAndGrads(targetLogProbFn, state) {
  const { values, keys } = stateToArray(state)
  const wrappedFn = wrapTargetLogProbFn(targetLogProbFn, keys)

  // Create variables for gradient computation
  const vars = values.map((v, i) => {
    const variable = tf.variable(v, true, `__state_${i}`)
    return variable
  })

  const { value, grads: gradMap } = tf.variableGrads(
    () => wrappedFn(...vars)
  )

  // Extract gradients in order
  const gradsArray = vars.map(v => gradMap[v.name].clone())

  // Clean up variables
  for (const v of vars) v.dispose()
  for (const g of Object.values(gradMap)) g.dispose()

  return {
    value,
    grads: arrayToState(gradsArray, keys)
  }
}
