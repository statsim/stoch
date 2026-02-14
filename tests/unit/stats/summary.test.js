import { summary } from '../../../src/stats/summary'

describe('summary', () => {
  test('computes correct stats for a single chain', () => {
    // Known samples
    const samples = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    const result = summary({ x: samples })

    expect(result.x.mean).toBeCloseTo(5.5, 5)
    // sample variance = sum((x-5.5)^2)/9 = 82.5/9 = 9.1667
    expect(result.x.sd).toBeCloseTo(Math.sqrt(82.5 / 9), 3)
    expect(result.x.hdiLow).toBeDefined()
    expect(result.x.hdiHigh).toBeDefined()
    expect(result.x.hdiLow).toBeLessThan(result.x.hdiHigh)
    expect(result.x.ess).toBeGreaterThan(0)
    expect(isNaN(result.x.rhat)).toBe(true) // single chain → no R-hat
    expect(result.x.mcse).toBeGreaterThan(0)
  })

  test('computes R-hat with multiple chains', () => {
    // Two similar chains from the same distribution
    const chain1 = Array.from({ length: 200 }, () => Math.random())
    const chain2 = Array.from({ length: 200 }, () => Math.random())
    const result = summary({ x: [chain1, chain2] })

    expect(result.x.mean).toBeGreaterThan(0.3)
    expect(result.x.mean).toBeLessThan(0.7)
    expect(result.x.rhat).toBeGreaterThan(0.9)
    expect(result.x.rhat).toBeLessThan(1.2)
  })

  test('handles multiple parameters', () => {
    const result = summary({
      mu: Array.from({ length: 100 }, () => Math.random()),
      sigma: Array.from({ length: 100 }, () => Math.random() + 1)
    })

    expect(result.mu).toBeDefined()
    expect(result.sigma).toBeDefined()
    expect(result.mu.mean).toBeGreaterThan(0)
    expect(result.sigma.mean).toBeGreaterThan(1)
  })

  test('custom HDI prob', () => {
    const samples = Array.from({ length: 1000 }, () => Math.random())
    const r50 = summary({ x: samples }, { hdiProb: 0.5 })
    const r99 = summary({ x: samples }, { hdiProb: 0.99 })

    // 50% HDI should be narrower than 99%
    const width50 = r50.x.hdiHigh - r50.x.hdiLow
    const width99 = r99.x.hdiHigh - r99.x.hdiLow
    expect(width50).toBeLessThan(width99)
  })

  test('ESS and MCSE are reasonable for iid samples', () => {
    const n = 500
    const samples = Array.from({ length: n }, () => Math.random())
    const result = summary({ x: samples })

    // For iid samples, ESS should be close to n
    expect(result.x.ess).toBeGreaterThan(n * 0.5)
    // MCSE should be small relative to sd
    expect(result.x.mcse).toBeLessThan(result.x.sd)
  })
})
