import * as tf from '@tensorflow/tfjs'
import { Tanh } from '../../../src/bijectors/tanh'

describe('Tanh bijector', () => {
  test('forward = tanh', () => {
    const b = new Tanh()
    const y = b.forward(tf.tensor([0, 1, -1]))
    const expected = [Math.tanh(0), Math.tanh(1), Math.tanh(-1)]
    const data = y.dataSync()
    for (let i = 0; i < 3; i++) {
      expect(data[i]).toBeCloseTo(expected[i], 5)
    }
    y.dispose()
  })

  test('inverse = atanh', () => {
    const b = new Tanh()
    const x = b.inverse(tf.tensor([0, 0.5, -0.5]))
    const expected = [Math.atanh(0), Math.atanh(0.5), Math.atanh(-0.5)]
    const data = x.dataSync()
    for (let i = 0; i < 3; i++) {
      expect(data[i]).toBeCloseTo(expected[i], 4)
    }
    x.dispose()
  })

  test('forward-inverse roundtrip', () => {
    const b = new Tanh()
    const x = tf.tensor([-2, -1, 0, 1, 2])
    const y = b.forward(x)
    const xBack = b.inverse(y)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 4)
    }
    x.dispose(); y.dispose(); xBack.dispose()
  })

  test('FLDJ + ILDJ = 0', () => {
    const b = new Tanh()
    const x = tf.tensor([-1, 0, 0.5, 1])
    const y = b.forward(x)
    const fldj = b.forwardLogDetJacobian(x)
    const ildj = b.inverseLogDetJacobian(y)
    const sum = tf.add(fldj, ildj).dataSync()
    for (let i = 0; i < sum.length; i++) {
      expect(sum[i]).toBeCloseTo(0, 4)
    }
    x.dispose(); y.dispose(); fldj.dispose(); ildj.dispose()
  })

  test('FLDJ at 0 = log(1)', () => {
    const b = new Tanh()
    const fldj = b.forwardLogDetJacobian(0)
    // tanh'(0) = 1 - tanh²(0) = 1, log(1) = 0
    expect(fldj.dataSync()[0]).toBeCloseTo(0, 5)
    fldj.dispose()
  })

  test('output in (-1, 1)', () => {
    const b = new Tanh()
    const y = b.forward(tf.tensor([-3, -1, 0, 1, 3]))
    const data = y.dataSync()
    for (let i = 0; i < data.length; i++) {
      expect(data[i]).toBeGreaterThan(-1)
      expect(data[i]).toBeLessThan(1)
    }
    y.dispose()
  })
})
