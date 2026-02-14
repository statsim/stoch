import { hdi } from '../../../src/stats/hdi'

describe('hdi', () => {
  test('returns interval containing the probability mass', () => {
    // Standard normal samples (approximately)
    const samples = []
    // Generate pseudo-normal via central limit theorem
    for (let i = 0; i < 10000; i++) {
      let s = 0
      for (let j = 0; j < 12; j++) s += Math.random()
      samples.push(s - 6) // approx N(0,1)
    }
    const [low, high] = hdi(samples, 0.94)
    // 94% HDI of N(0,1) should be roughly [-1.88, 1.88]
    expect(low).toBeLessThan(-1.0)
    expect(high).toBeGreaterThan(1.0)
    expect(high - low).toBeLessThan(5)
  })

  test('HDI of uniform is the narrowest window', () => {
    const samples = Array.from({ length: 1000 }, (_, i) => i / 999)
    const [low, high] = hdi(samples, 0.5)
    // 50% HDI of Uniform(0,1) should be about [0, 0.5] or similar 0.5-width window
    expect(high - low).toBeCloseTo(0.5, 1)
  })

  test('HDI of bimodal skews to denser mode', () => {
    // 800 samples near 0, 200 near 10
    const samples = []
    for (let i = 0; i < 800; i++) samples.push(i * 0.01)
    for (let i = 0; i < 200; i++) samples.push(10 + i * 0.01)
    const [low, high] = hdi(samples, 0.5)
    // Should pick the dense cluster near 0
    expect(high).toBeLessThan(9)
  })

  test('90% HDI is narrower than 99% HDI', () => {
    const samples = Array.from({ length: 1000 }, () => Math.random())
    const [low90, high90] = hdi(samples, 0.9)
    const [low99, high99] = hdi(samples, 0.99)
    expect(high90 - low90).toBeLessThan(high99 - low99)
  })

  test('throws for invalid prob', () => {
    expect(() => hdi([1, 2, 3], 0)).toThrow()
    expect(() => hdi([1, 2, 3], 1)).toThrow()
    expect(() => hdi([1, 2, 3], -0.5)).toThrow()
  })

  test('throws for too few samples', () => {
    expect(() => hdi([1])).toThrow()
    expect(() => hdi([])).toThrow()
  })

  test('works with Float32Array', () => {
    const samples = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    const [low, high] = hdi(samples, 0.5)
    expect(high - low).toBeLessThanOrEqual(5)
    expect(low).toBeGreaterThanOrEqual(1)
    expect(high).toBeLessThanOrEqual(10)
  })
})
