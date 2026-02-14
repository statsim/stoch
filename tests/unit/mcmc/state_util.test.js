import * as tf from '@tensorflow/tfjs'
import {
  stateToArray,
  arrayToState,
  cloneState,
  disposeState,
  computeGrads,
  valueAndGrads
} from '../../../src/mcmc/state_util'
import { expectClose } from '../../helpers/tolerance'

describe('MCMC state utilities', () => {
  describe('stateToArray / arrayToState', () => {
    test('single tensor roundtrip', () => {
      const state = tf.tensor([1, 2, 3])
      const { values, keys } = stateToArray(state)
      expect(keys).toBeNull()
      expect(values.length).toBe(1)
      expect(values[0]).toBe(state)

      const restored = arrayToState(values, keys)
      expect(restored).toBe(state)
      state.dispose()
    })

    test('object state roundtrip', () => {
      const state = {
        mu: tf.scalar(1),
        sigma: tf.scalar(2)
      }
      const { values, keys } = stateToArray(state)
      expect(keys).toEqual(['mu', 'sigma'])
      expect(values.length).toBe(2)
      expect(values[0].dataSync()[0]).toBe(1)
      expect(values[1].dataSync()[0]).toBe(2)

      const restored = arrayToState(values, keys)
      expect(restored.mu.dataSync()[0]).toBe(1)
      expect(restored.sigma.dataSync()[0]).toBe(2)
      state.mu.dispose()
      state.sigma.dispose()
    })
  })

  describe('cloneState', () => {
    test('clones single tensor', () => {
      const state = tf.scalar(5)
      const cloned = cloneState(state)
      expect(cloned).not.toBe(state)
      expect(cloned.dataSync()[0]).toBe(5)
      state.dispose()
      cloned.dispose()
    })

    test('clones object state', () => {
      const state = { a: tf.scalar(1), b: tf.tensor([2, 3]) }
      const cloned = cloneState(state)
      expect(cloned.a).not.toBe(state.a)
      expect(cloned.b).not.toBe(state.b)
      expect(cloned.a.dataSync()[0]).toBe(1)
      expect(Array.from(cloned.b.dataSync())).toEqual([2, 3])
      disposeState(state)
      disposeState(cloned)
    })
  })

  describe('disposeState', () => {
    test('disposes single tensor', () => {
      const before = tf.memory().numTensors
      const state = tf.scalar(5)
      expect(tf.memory().numTensors).toBe(before + 1)
      disposeState(state)
      expect(tf.memory().numTensors).toBe(before)
    })

    test('disposes object state', () => {
      const before = tf.memory().numTensors
      const state = { a: tf.scalar(1), b: tf.scalar(2) }
      expect(tf.memory().numTensors).toBe(before + 2)
      disposeState(state)
      expect(tf.memory().numTensors).toBe(before)
    })

    test('safe to call on already-disposed tensors', () => {
      const state = tf.scalar(5)
      state.dispose()
      expect(() => disposeState(state)).not.toThrow()
    })
  })

  describe('computeGrads', () => {
    test('scalar state gradient', () => {
      // f(x) = x² → df/dx = 2x
      const targetLogProb = (x) => tf.square(x)
      const state = tf.scalar(3)

      const { value, grads } = computeGrads(targetLogProb, state)
      expectClose(value.dataSync()[0], 9, { atol: 1e-5 })
      expectClose(grads.dataSync()[0], 6, { atol: 1e-5 })

      value.dispose()
      grads.dispose()
      state.dispose()
    })

    test('object state gradient', () => {
      // f({a, b}) = a² + 2*b → df/da = 2a, df/db = 2
      const targetLogProb = ({ a, b }) => tf.add(tf.square(a), tf.mul(2, b))
      const state = { a: tf.scalar(3), b: tf.scalar(5) }

      const { value, grads } = computeGrads(targetLogProb, state)
      expectClose(value.dataSync()[0], 19, { atol: 1e-5 })
      expectClose(grads.a.dataSync()[0], 6, { atol: 1e-5 })
      expectClose(grads.b.dataSync()[0], 2, { atol: 1e-5 })

      value.dispose()
      disposeState(grads)
      disposeState(state)
    })

    test('vector state gradient', () => {
      // f(x) = sum(x²) where x is [2, 3] → grad = [4, 6]
      const targetLogProb = (x) => tf.sum(tf.square(x))
      const state = tf.tensor([2, 3])

      const { value, grads } = computeGrads(targetLogProb, state)
      expectClose(value.dataSync()[0], 13, { atol: 1e-5 })
      expect(Array.from(grads.dataSync())).toEqual(
        expect.arrayContaining([])
      )
      expectClose(grads.dataSync()[0], 4, { atol: 1e-5 })
      expectClose(grads.dataSync()[1], 6, { atol: 1e-5 })

      value.dispose()
      grads.dispose()
      state.dispose()
    })
  })

  describe('valueAndGrads', () => {
    test('scalar state', () => {
      const targetLogProb = (x) => tf.square(x)
      const state = tf.scalar(3)

      const { value, grads } = valueAndGrads(targetLogProb, state)
      expectClose(value.dataSync()[0], 9, { atol: 1e-5 })
      expectClose(grads.dataSync()[0], 6, { atol: 1e-5 })

      value.dispose()
      grads.dispose()
      state.dispose()
    })

    test('object state', () => {
      const targetLogProb = ({ a, b }) => tf.add(tf.square(a), tf.mul(2, b))
      const state = { a: tf.scalar(3), b: tf.scalar(5) }

      const { value, grads } = valueAndGrads(targetLogProb, state)
      expectClose(value.dataSync()[0], 19, { atol: 1e-5 })
      expectClose(grads.a.dataSync()[0], 6, { atol: 1e-5 })
      expectClose(grads.b.dataSync()[0], 2, { atol: 1e-5 })

      value.dispose()
      disposeState(grads)
      disposeState(state)
    })
  })
})
