import * as tf from '@tensorflow/tfjs'
import { CorrelationCholesky } from '../../../src/bijectors/correlation_cholesky'

describe('CorrelationCholesky bijector', () => {
  test('forward produces valid Cholesky factor (d=2)', () => {
    const b = new CorrelationCholesky()
    // d=2, k=1: single partial correlation
    const L = b.forward(tf.tensor([0]))
    expect(L.shape).toEqual([2, 2])
    const data = L.dataSync()
    // L[0,0] = 1
    expect(data[0]).toBeCloseTo(1, 5)
    // L[0,1] = 0
    expect(data[1]).toBeCloseTo(0, 5)
    // L[1,0] = tanh(0) = 0
    expect(data[2]).toBeCloseTo(0, 4)
    // L[1,1] = sqrt(1 - 0²) = 1
    expect(data[3]).toBeCloseTo(1, 4)
    L.dispose()
  })

  test('forward: L*L^T is a correlation matrix', () => {
    const b = new CorrelationCholesky()
    const L = b.forward(tf.tensor([1, -0.5, 0.3]))
    expect(L.shape).toEqual([3, 3])

    // Compute R = L * L^T
    const R = tf.matMul(L, tf.transpose(L))
    const rData = R.dataSync()

    // Diagonal should be 1
    expect(rData[0]).toBeCloseTo(1, 3)  // R[0,0]
    expect(rData[4]).toBeCloseTo(1, 3)  // R[1,1]
    expect(rData[8]).toBeCloseTo(1, 3)  // R[2,2]

    // Off-diagonal should be in (-1, 1)
    expect(Math.abs(rData[1])).toBeLessThan(1)
    expect(Math.abs(rData[2])).toBeLessThan(1)
    expect(Math.abs(rData[5])).toBeLessThan(1)

    // Symmetric
    expect(rData[1]).toBeCloseTo(rData[3], 5)
    expect(rData[2]).toBeCloseTo(rData[6], 5)
    expect(rData[5]).toBeCloseTo(rData[7], 5)

    L.dispose(); R.dispose()
  })

  test('L is lower triangular', () => {
    const b = new CorrelationCholesky()
    const L = b.forward(tf.tensor([1, -0.5, 0.3]))
    const data = L.dataSync()
    // Upper triangle should be 0
    expect(data[1]).toBeCloseTo(0, 5)  // L[0,1]
    expect(data[2]).toBeCloseTo(0, 5)  // L[0,2]
    expect(data[5]).toBeCloseTo(0, 5)  // L[1,2]
    L.dispose()
  })

  test('forward-inverse roundtrip', () => {
    const b = new CorrelationCholesky()
    const x = tf.tensor([0.5, -1, 0.8])
    const L = b.forward(x)
    const xBack = b.inverse(L)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 3)
    }
    x.dispose(); L.dispose(); xBack.dispose()
  })

  test('FLDJ + ILDJ = 0', () => {
    const b = new CorrelationCholesky()
    const x = tf.tensor([0.5, -0.3, 0.8])
    const y = b.forward(x)
    const fldj = b.forwardLogDetJacobian(x, 1)
    const ildj = b.inverseLogDetJacobian(y, 2)
    const sum = fldj.dataSync()[0] + ildj.dataSync()[0]
    expect(sum).toBeCloseTo(0, 2)
    x.dispose(); y.dispose(); fldj.dispose(); ildj.dispose()
  })

  test('d=2 specific values', () => {
    const b = new CorrelationCholesky()
    // x=1 → tanh(1) ≈ 0.7616
    const L = b.forward(tf.tensor([1]))
    const data = L.dataSync()
    const rho = Math.tanh(1)
    expect(data[2]).toBeCloseTo(rho, 3)  // L[1,0]
    expect(data[3]).toBeCloseTo(Math.sqrt(1 - rho * rho), 3)  // L[1,1]
    L.dispose()
  })
})
