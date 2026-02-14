import { mcse } from '../../../src/stats/mcse'

describe('mcse', () => {
  test('MCSE of iid samples ≈ sd/sqrt(n)', () => {
    // IID samples have ESS ≈ n, so MCSE ≈ sd/sqrt(n)
    const n = 1000
    const samples = Array.from({ length: n }, () => Math.random())
    let sum = 0
    for (let i = 0; i < n; i++) sum += samples[i]
    const mean = sum / n
    let varSum = 0
    for (let i = 0; i < n; i++) {
      const d = samples[i] - mean
      varSum += d * d
    }
    const sd = Math.sqrt(varSum / (n - 1))
    const expected = sd / Math.sqrt(n)
    const result = mcse(samples)
    // Should be close to sd/sqrt(n) for iid samples
    expect(result).toBeGreaterThan(expected * 0.5)
    expect(result).toBeLessThan(expected * 2)
  })

  test('MCSE of autocorrelated samples > iid MCSE', () => {
    // Random walk has high autocorrelation → ESS < n → MCSE > sd/sqrt(n)
    const n = 500
    const samples = [0]
    for (let i = 1; i < n; i++) {
      samples.push(samples[i - 1] + (Math.random() - 0.5) * 0.1)
    }
    let sum = 0
    for (let i = 0; i < n; i++) sum += samples[i]
    const mean = sum / n
    let varSum = 0
    for (let i = 0; i < n; i++) {
      const d = samples[i] - mean
      varSum += d * d
    }
    const sd = Math.sqrt(varSum / (n - 1))
    const iidMcse = sd / Math.sqrt(n)
    const result = mcse(samples)
    // Autocorrelated → MCSE should be larger than iid estimate
    expect(result).toBeGreaterThan(iidMcse)
  })

  test('MCSE is positive', () => {
    const samples = Array.from({ length: 100 }, () => Math.random() * 10)
    expect(mcse(samples)).toBeGreaterThan(0)
  })

  test('throws for too few samples', () => {
    expect(() => mcse([1])).toThrow()
    expect(() => mcse([])).toThrow()
  })
})
