import * as tf from '@tensorflow/tfjs'
import { ndtri } from '../../../src/math/ndtri'
import { ndtr } from '../../../src/math/special'
import { expectClose } from '../../helpers/tolerance'

describe('math/ndtri', () => {
  // Reference values from scipy.stats.norm.ppf
  const cases = [
    [0.5, 0.0],
    [0.8413447, 1.0],
    [0.1586553, -1.0],
    [0.9772499, 2.0],
    [0.0227501, -2.0],
    [0.9986501, 3.0],
    [0.0013499, -3.0],
    [0.025, -1.959964],
    [0.975, 1.959964],
    [0.01, -2.326348],
    [0.99, 2.326348],
    [0.001, -3.090232],
    [0.999, 3.090232],
  ]

  cases.forEach(([p, expected]) => {
    test(`ndtri(${p}) ≈ ${expected}`, () => {
      const result = ndtri(p)
      expectClose(result.dataSync()[0], expected, { rtol: 1e-3, atol: 1e-3 })
      result.dispose()
    })
  })

  test('works element-wise on tensors', () => {
    const p = tf.tensor([0.1, 0.25, 0.5, 0.75, 0.9])
    const result = ndtri(p)
    const data = result.dataSync()
    // ndtri(0.5) = 0
    expectClose(data[2], 0, { atol: 1e-5 })
    // Symmetry: ndtri(p) = -ndtri(1-p)
    expectClose(data[0] + data[4], 0, { atol: 1e-3 })
    expectClose(data[1] + data[3], 0, { atol: 1e-3 })
    result.dispose()
    p.dispose()
  })

  test('roundtrip: ndtr(ndtri(p)) ≈ p', () => {
    const p = tf.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    const x = ndtri(p)
    const recovered = ndtr(x)
    const pData = p.dataSync()
    const rData = recovered.dataSync()
    for (let i = 0; i < pData.length; i++) {
      expectClose(rData[i], pData[i], { rtol: 1e-3, atol: 1e-4 })
    }
    p.dispose()
    x.dispose()
    recovered.dispose()
  })

  test('handles tail probabilities', () => {
    const p = tf.tensor([0.001, 0.005, 0.995, 0.999])
    const result = ndtri(p)
    const data = result.dataSync()
    // All should be finite
    for (let i = 0; i < data.length; i++) {
      expect(isFinite(data[i])).toBe(true)
    }
    // Lower tail should be negative, upper tail positive
    expect(data[0]).toBeLessThan(0)
    expect(data[1]).toBeLessThan(0)
    expect(data[2]).toBeGreaterThan(0)
    expect(data[3]).toBeGreaterThan(0)
    result.dispose()
    p.dispose()
  })
})
