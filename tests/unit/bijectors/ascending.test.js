import * as tf from '@tensorflow/tfjs'
import { Ascending } from '../../../src/bijectors/ascending'

describe('Ascending bijector', () => {
  test('forward produces sorted output', () => {
    const b = new Ascending()
    const y = b.forward(tf.tensor([0, 1, 0.5, 2]))
    const data = y.dataSync()
    for (let i = 1; i < data.length; i++) {
      expect(data[i]).toBeGreaterThan(data[i - 1])
    }
    y.dispose()
  })

  test('forward first element = input first element', () => {
    const b = new Ascending()
    const y = b.forward(tf.tensor([3, 1, 2]))
    expect(y.dataSync()[0]).toBeCloseTo(3, 5)
    y.dispose()
  })

  test('forward-inverse roundtrip', () => {
    const b = new Ascending()
    const x = tf.tensor([0, 1, -0.5, 2, 0.3])
    const y = b.forward(x)
    const xBack = b.inverse(y)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 4)
    }
    x.dispose(); y.dispose(); xBack.dispose()
  })

  test('FLDJ with eventNdims=1', () => {
    const b = new Ascending()
    const x = tf.tensor([0, 1, 2])
    const fldj = b.forwardLogDetJacobian(x, 1)
    // FLDJ = sum of log(sigmoid(x[i])) for i > 0
    const expected = (1 - Math.log(1 + Math.exp(1))) + (2 - Math.log(1 + Math.exp(2)))
    expect(fldj.dataSync()[0]).toBeCloseTo(expected, 4)
    x.dispose(); fldj.dispose()
  })

  test('FLDJ + ILDJ = 0 (with eventNdims=1)', () => {
    const b = new Ascending()
    const x = tf.tensor([0, 1, -0.5, 2])
    const y = b.forward(x)
    const fldj = b.forwardLogDetJacobian(x, 1)
    const ildj = b.inverseLogDetJacobian(y, 1)
    const sum = fldj.dataSync()[0] + ildj.dataSync()[0]
    expect(sum).toBeCloseTo(0, 3)
    x.dispose(); y.dispose(); fldj.dispose(); ildj.dispose()
  })

  test('batched forward', () => {
    const b = new Ascending()
    const x = tf.tensor([[0, 1, 2], [1, 0, 1]])
    const y = b.forward(x)
    expect(y.shape).toEqual([2, 3])
    const data = y.dataSync()
    // Each row should be sorted
    expect(data[1]).toBeGreaterThan(data[0])
    expect(data[2]).toBeGreaterThan(data[1])
    expect(data[4]).toBeGreaterThan(data[3])
    expect(data[5]).toBeGreaterThan(data[4])
    y.dispose()
  })

  test('single element is identity', () => {
    const b = new Ascending()
    const y = b.forward(tf.tensor([5]))
    expect(y.dataSync()[0]).toBeCloseTo(5, 5)
    y.dispose()
  })
})
