import * as tf from '@tensorflow/tfjs'
import { FillTriangular } from '../../../src/bijectors/fill_triangular'

describe('FillTriangular bijector', () => {
  test('forward: vec to lower triangular 2x2', () => {
    const b = new FillTriangular()
    // n=2, k=3: [L00, L10, L11]
    const y = b.forward(tf.tensor([1, 2, 3]))
    expect(y.shape).toEqual([2, 2])
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(1, 5)  // L[0,0]
    expect(data[1]).toBeCloseTo(0, 5)  // L[0,1] = 0
    expect(data[2]).toBeCloseTo(2, 5)  // L[1,0]
    expect(data[3]).toBeCloseTo(3, 5)  // L[1,1]
    y.dispose()
  })

  test('forward: vec to lower triangular 3x3', () => {
    const b = new FillTriangular()
    // n=3, k=6: [L00, L10, L11, L20, L21, L22]
    const y = b.forward(tf.tensor([1, 2, 3, 4, 5, 6]))
    expect(y.shape).toEqual([3, 3])
    const data = y.dataSync()
    // Row 0: [1, 0, 0]
    expect(data[0]).toBeCloseTo(1, 5)
    expect(data[1]).toBeCloseTo(0, 5)
    expect(data[2]).toBeCloseTo(0, 5)
    // Row 1: [2, 3, 0]
    expect(data[3]).toBeCloseTo(2, 5)
    expect(data[4]).toBeCloseTo(3, 5)
    expect(data[5]).toBeCloseTo(0, 5)
    // Row 2: [4, 5, 6]
    expect(data[6]).toBeCloseTo(4, 5)
    expect(data[7]).toBeCloseTo(5, 5)
    expect(data[8]).toBeCloseTo(6, 5)
    y.dispose()
  })

  test('inverse recovers vector', () => {
    const b = new FillTriangular()
    const x = tf.tensor([1, 2, 3, 4, 5, 6])
    const y = b.forward(x)
    const xBack = b.inverse(y)
    expect(xBack.shape).toEqual([6])
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 5)
    }
    x.dispose(); y.dispose(); xBack.dispose()
  })

  test('forward-inverse roundtrip with batch', () => {
    const b = new FillTriangular()
    const x = tf.tensor([[1, 2, 3], [4, 5, 6]])
    const y = b.forward(x)
    expect(y.shape).toEqual([2, 2, 2])
    const xBack = b.inverse(y)
    expect(xBack.shape).toEqual([2, 3])
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 5)
    }
    x.dispose(); y.dispose(); xBack.dispose()
  })

  test('log-det-Jacobian is 0', () => {
    const b = new FillTriangular()
    const fldj = b.forwardLogDetJacobian(tf.tensor([1, 2, 3, 4, 5, 6]), 1)
    expect(fldj.dataSync()[0]).toBeCloseTo(0, 5)
    fldj.dispose()
  })

  test('isConstantJacobian = true', () => {
    const b = new FillTriangular()
    expect(b.isConstantJacobian).toBe(true)
  })
})
