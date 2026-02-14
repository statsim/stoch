import * as tf from '@tensorflow/tfjs'
import { Bijector } from '../../../src/bijectors/bijector'

describe('Bijector base class', () => {
  describe('constructor', () => {
    test('default properties', () => {
      const b = new Bijector()
      expect(b.name).toBe('Bijector')
      expect(b.forwardMinEventNdims).toBe(0)
      expect(b.inverseMinEventNdims).toBe(0)
      expect(b.isConstantJacobian).toBe(false)
    })

    test('custom properties', () => {
      const b = new Bijector({
        forwardMinEventNdims: 1,
        inverseMinEventNdims: 1,
        isConstantJacobian: true,
        name: 'TestBijector'
      })
      expect(b.name).toBe('TestBijector')
      expect(b.forwardMinEventNdims).toBe(1)
      expect(b.inverseMinEventNdims).toBe(1)
      expect(b.isConstantJacobian).toBe(true)
    })
  })

  describe('abstract methods throw', () => {
    const b = new Bijector({ name: 'TestBij' })

    test('_forward throws', () => {
      expect(() => b.forward(1)).toThrow('TestBij._forward not implemented')
    })

    test('_inverse throws', () => {
      expect(() => b.inverse(1)).toThrow('TestBij._inverse not implemented')
    })

    test('_forwardLogDetJacobian throws', () => {
      expect(() => b.forwardLogDetJacobian(1)).toThrow('TestBij._forwardLogDetJacobian not implemented')
    })

    test('_inverseLogDetJacobian throws', () => {
      expect(() => b.inverseLogDetJacobian(1)).toThrow('TestBij._inverseLogDetJacobian not implemented')
    })
  })

  describe('concrete subclass', () => {
    // Simple doubling bijector for testing
    class DoubleBijector extends Bijector {
      constructor() {
        super({ isConstantJacobian: true, name: 'Double' })
      }
      _forward(x) { return tf.mul(x, 2) }
      _inverse(y) { return tf.div(y, 2) }
      _forwardLogDetJacobian(x) { return tf.scalar(Math.log(2)) }
      _inverseLogDetJacobian(y) { return tf.scalar(-Math.log(2)) }
    }

    test('forward works with scalars', () => {
      const b = new DoubleBijector()
      const y = b.forward(3)
      expect(y.dataSync()[0]).toBeCloseTo(6)
      y.dispose()
    })

    test('forward works with tensors', () => {
      const b = new DoubleBijector()
      const y = b.forward(tf.tensor([1, 2, 3]))
      expect(Array.from(y.dataSync())).toEqual([2, 4, 6])
      y.dispose()
    })

    test('inverse works', () => {
      const b = new DoubleBijector()
      const x = b.inverse(6)
      expect(x.dataSync()[0]).toBeCloseTo(3)
      x.dispose()
    })

    test('forward-inverse roundtrip', () => {
      const b = new DoubleBijector()
      const x = tf.tensor([1, 2, 3, 4, 5])
      const y = b.forward(x)
      const xBack = b.inverse(y)
      const xData = x.dataSync()
      const xBackData = xBack.dataSync()
      for (let i = 0; i < xData.length; i++) {
        expect(xBackData[i]).toBeCloseTo(xData[i], 5)
      }
      x.dispose()
      y.dispose()
      xBack.dispose()
    })

    test('forwardLogDetJacobian', () => {
      const b = new DoubleBijector()
      const ldj = b.forwardLogDetJacobian(1)
      expect(ldj.dataSync()[0]).toBeCloseTo(Math.log(2))
      ldj.dispose()
    })

    test('inverseLogDetJacobian', () => {
      const b = new DoubleBijector()
      const ldj = b.inverseLogDetJacobian(1)
      expect(ldj.dataSync()[0]).toBeCloseTo(-Math.log(2))
      ldj.dispose()
    })

    test('FLDJ + ILDJ = 0 (inverse function theorem)', () => {
      const b = new DoubleBijector()
      const x = tf.scalar(5)
      const y = b.forward(x)
      const fldj = b.forwardLogDetJacobian(x)
      const ildj = b.inverseLogDetJacobian(y)
      expect(fldj.dataSync()[0] + ildj.dataSync()[0]).toBeCloseTo(0, 5)
      x.dispose()
      y.dispose()
      fldj.dispose()
      ildj.dispose()
    })
  })

  describe('event dimension reduction', () => {
    // Bijector with per-element log-det-Jacobian
    class ElementWiseBijector extends Bijector {
      constructor() {
        super({ name: 'ElementWise' })
      }
      _forward(x) { return tf.mul(x, 2) }
      _inverse(y) { return tf.div(y, 2) }
      _forwardLogDetJacobian(x) {
        return tf.fill(x.shape, Math.log(2))
      }
      _inverseLogDetJacobian(y) {
        return tf.fill(y.shape, -Math.log(2))
      }
    }

    test('eventNdims=0 returns per-element FLDJ', () => {
      const b = new ElementWiseBijector()
      const ldj = b.forwardLogDetJacobian(tf.tensor([1, 2, 3]), 0)
      expect(ldj.shape).toEqual([3])
      const data = ldj.dataSync()
      for (let i = 0; i < 3; i++) {
        expect(data[i]).toBeCloseTo(Math.log(2))
      }
      ldj.dispose()
    })

    test('eventNdims=1 sums over last dim', () => {
      const b = new ElementWiseBijector()
      const ldj = b.forwardLogDetJacobian(tf.tensor([1, 2, 3]), 1)
      expect(ldj.shape).toEqual([])
      expect(ldj.dataSync()[0]).toBeCloseTo(3 * Math.log(2))
      ldj.dispose()
    })

    test('eventNdims=1 with batched input', () => {
      const b = new ElementWiseBijector()
      const x = tf.tensor([[1, 2], [3, 4]])
      const ldj = b.forwardLogDetJacobian(x, 1)
      expect(ldj.shape).toEqual([2])
      const data = ldj.dataSync()
      expect(data[0]).toBeCloseTo(2 * Math.log(2))
      expect(data[1]).toBeCloseTo(2 * Math.log(2))
      ldj.dispose()
      x.dispose()
    })
  })

  describe('tf.tidy memory management', () => {
    class SimpleBijector extends Bijector {
      constructor() {
        super({ name: 'Simple' })
      }
      _forward(x) { return tf.add(x, 1) }
      _inverse(y) { return tf.sub(y, 1) }
      _forwardLogDetJacobian(x) { return tf.scalar(0) }
      _inverseLogDetJacobian(y) { return tf.scalar(0) }
    }

    test('forward does not leak tensors', () => {
      const b = new SimpleBijector()
      const input = tf.scalar(5)
      const before = tf.memory().numTensors
      const y = b.forward(input)
      y.dispose()
      expect(tf.memory().numTensors).toBe(before)
      input.dispose()
    })

    test('inverse does not leak tensors', () => {
      const b = new SimpleBijector()
      const input = tf.scalar(5)
      const before = tf.memory().numTensors
      const x = b.inverse(input)
      x.dispose()
      expect(tf.memory().numTensors).toBe(before)
      input.dispose()
    })
  })

  test('toString', () => {
    const b = new Bijector({ name: 'MyBij' })
    expect(b.toString()).toBe('MyBij()')
  })
})
