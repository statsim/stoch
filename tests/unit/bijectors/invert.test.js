import * as tf from '@tensorflow/tfjs'
import { Invert } from '../../../src/bijectors/invert'
import { Exp } from '../../../src/bijectors/exp'
import { Sigmoid } from '../../../src/bijectors/sigmoid'

describe('Invert bijector', () => {
  test('Invert(Exp) forward = log', () => {
    const b = new Invert({ bijector: new Exp() })
    const y = b.forward(tf.tensor([1, Math.E, Math.E * Math.E]))
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(0, 5)
    expect(data[1]).toBeCloseTo(1, 5)
    expect(data[2]).toBeCloseTo(2, 4)
    y.dispose()
    b.dispose()
  })

  test('Invert(Exp) inverse = exp', () => {
    const b = new Invert({ bijector: new Exp() })
    const x = b.inverse(tf.tensor([0, 1, 2]))
    const data = x.dataSync()
    expect(data[0]).toBeCloseTo(1, 5)
    expect(data[1]).toBeCloseTo(Math.E, 4)
    expect(data[2]).toBeCloseTo(Math.E * Math.E, 3)
    x.dispose()
    b.dispose()
  })

  test('forward-inverse roundtrip', () => {
    const b = new Invert({ bijector: new Sigmoid() })
    const x = tf.tensor([0.1, 0.5, 0.9])
    const y = b.forward(x)
    const xBack = b.inverse(y)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 4)
    }
    x.dispose(); y.dispose(); xBack.dispose()
    b.dispose()
  })

  test('FLDJ + ILDJ = 0', () => {
    const inner = new Exp()
    const b = new Invert({ bijector: inner })
    const x = tf.tensor([1, 2, 3])
    const y = b.forward(x)
    const fldj = b.forwardLogDetJacobian(x)
    const ildj = b.inverseLogDetJacobian(y)
    const sum = tf.add(fldj, ildj).dataSync()
    for (let i = 0; i < sum.length; i++) {
      expect(sum[i]).toBeCloseTo(0, 4)
    }
    x.dispose(); y.dispose(); fldj.dispose(); ildj.dispose()
    b.dispose()
  })

  test('name includes inner bijector', () => {
    const b = new Invert({ bijector: new Exp() })
    expect(b.name).toBe('Invert(Exp)')
    b.dispose()
  })

  test('preserves event ndims', () => {
    const inner = new Exp()
    const b = new Invert({ bijector: inner })
    expect(b.forwardMinEventNdims).toBe(inner.inverseMinEventNdims)
    expect(b.inverseMinEventNdims).toBe(inner.forwardMinEventNdims)
    b.dispose()
  })
})
