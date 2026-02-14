import * as tf from '@tensorflow/tfjs'
import { AffineScalar } from '../../../src/bijectors/affine_scalar'

describe('AffineScalar bijector', () => {
  test('forward: shift + scale * x', () => {
    const b = new AffineScalar({ shift: 1, scale: 2 })
    const y = b.forward(tf.tensor([0, 1, 2]))
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(1, 5)
    expect(data[1]).toBeCloseTo(3, 5)
    expect(data[2]).toBeCloseTo(5, 5)
    y.dispose()
    b.dispose()
  })

  test('inverse: (y - shift) / scale', () => {
    const b = new AffineScalar({ shift: 1, scale: 2 })
    const x = b.inverse(tf.tensor([1, 3, 5]))
    const data = x.dataSync()
    expect(data[0]).toBeCloseTo(0, 5)
    expect(data[1]).toBeCloseTo(1, 5)
    expect(data[2]).toBeCloseTo(2, 5)
    x.dispose()
    b.dispose()
  })

  test('forward-inverse roundtrip', () => {
    const b = new AffineScalar({ shift: -3, scale: 0.5 })
    const x = tf.tensor([0, 1, 2, 10])
    const y = b.forward(x)
    const xBack = b.inverse(y)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 5)
    }
    x.dispose(); y.dispose(); xBack.dispose()
    b.dispose()
  })

  test('FLDJ = log|scale|', () => {
    const b = new AffineScalar({ shift: 5, scale: 3 })
    const fldj = b.forwardLogDetJacobian(tf.tensor([0, 1, 2]))
    const data = fldj.dataSync()
    for (let i = 0; i < 3; i++) {
      expect(data[i]).toBeCloseTo(Math.log(3), 5)
    }
    fldj.dispose()
    b.dispose()
  })

  test('ILDJ = -log|scale|', () => {
    const b = new AffineScalar({ shift: 5, scale: 3 })
    const ildj = b.inverseLogDetJacobian(tf.tensor([0, 1, 2]))
    const data = ildj.dataSync()
    for (let i = 0; i < 3; i++) {
      expect(data[i]).toBeCloseTo(-Math.log(3), 5)
    }
    ildj.dispose()
    b.dispose()
  })

  test('isConstantJacobian = true', () => {
    const b = new AffineScalar({ shift: 1, scale: 2 })
    expect(b.isConstantJacobian).toBe(true)
    b.dispose()
  })

  test('default shift=0, scale=1 is identity', () => {
    const b = new AffineScalar()
    const y = b.forward(tf.tensor([1, 2, 3]))
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(1, 5)
    expect(data[1]).toBeCloseTo(2, 5)
    expect(data[2]).toBeCloseTo(3, 5)
    y.dispose()
    b.dispose()
  })
})
