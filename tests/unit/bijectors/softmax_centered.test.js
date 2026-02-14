import * as tf from '@tensorflow/tfjs'
import { SoftmaxCentered } from '../../../src/bijectors/softmax_centered'

describe('SoftmaxCentered bijector', () => {
  test('forward maps R^(d-1) to simplex', () => {
    const b = new SoftmaxCentered()
    const y = b.forward(tf.tensor([0, 0]))
    const data = y.dataSync()
    expect(y.shape).toEqual([3])
    // softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
    for (let i = 0; i < 3; i++) {
      expect(data[i]).toBeCloseTo(1 / 3, 4)
    }
    y.dispose()
  })

  test('forward output sums to 1', () => {
    const b = new SoftmaxCentered()
    const y = b.forward(tf.tensor([1, -1, 2]))
    const sum = tf.sum(y).dataSync()[0]
    expect(sum).toBeCloseTo(1, 5)
    y.dispose()
  })

  test('forward output all positive', () => {
    const b = new SoftmaxCentered()
    const y = b.forward(tf.tensor([-5, 0, 5]))
    const data = y.dataSync()
    for (let i = 0; i < data.length; i++) {
      expect(data[i]).toBeGreaterThan(0)
    }
    y.dispose()
  })

  test('inverse recovers input', () => {
    const b = new SoftmaxCentered()
    const x = tf.tensor([1, -1, 2])
    const y = b.forward(x)
    const xBack = b.inverse(y)
    expect(xBack.shape).toEqual([3])
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 4)
    }
    x.dispose(); y.dispose(); xBack.dispose()
  })

  test('FLDJ + ILDJ = 0 (with eventNdims=1)', () => {
    const b = new SoftmaxCentered()
    const x = tf.tensor([1, -1])
    const y = b.forward(x)
    const fldj = b.forwardLogDetJacobian(x, 1)
    const ildj = b.inverseLogDetJacobian(y, 1)
    const sum = fldj.dataSync()[0] + ildj.dataSync()[0]
    expect(sum).toBeCloseTo(0, 4)
    x.dispose(); y.dispose(); fldj.dispose(); ildj.dispose()
  })

  test('batched forward', () => {
    const b = new SoftmaxCentered()
    const x = tf.tensor([[0, 0], [1, -1]])
    const y = b.forward(x)
    expect(y.shape).toEqual([2, 3])
    // Each row sums to 1
    const sums = tf.sum(y, 1).dataSync()
    expect(sums[0]).toBeCloseTo(1, 5)
    expect(sums[1]).toBeCloseTo(1, 5)
    y.dispose()
  })

  test('dimension increases by 1', () => {
    const b = new SoftmaxCentered()
    const y = b.forward(tf.tensor([0, 0, 0, 0]))
    expect(y.shape).toEqual([5])
    y.dispose()
  })
})
