import { effectiveSampleSize, potentialScaleReduction } from '../../../src/mcmc/diagnostics'

describe('MCMC diagnostics', () => {
  describe('effectiveSampleSize', () => {
    test('returns N for iid samples', () => {
      // Generate pseudo-iid samples (uniform random)
      const n = 1000
      const samples = []
      // Simple LCG for reproducibility
      let seed = 12345
      for (let i = 0; i < n; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        samples.push(seed / 0x7fffffff)
      }

      const ess = effectiveSampleSize(samples)
      // ESS should be close to N for iid samples
      expect(ess).toBeGreaterThan(n * 0.5)
      expect(ess).toBeLessThanOrEqual(n)
    })

    test('returns low ESS for highly autocorrelated chain', () => {
      // Random walk: each sample depends heavily on previous
      const n = 1000
      const samples = [0]
      let seed = 54321
      for (let i = 1; i < n; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        const noise = (seed / 0x7fffffff - 0.5) * 0.1
        samples.push(samples[i - 1] + noise)
      }

      const ess = effectiveSampleSize(samples)
      // ESS should be much less than N for autocorrelated chain
      expect(ess).toBeLessThan(n * 0.3)
      expect(ess).toBeGreaterThan(0)
    })

    test('handles constant chain', () => {
      const samples = new Array(100).fill(5.0)
      const ess = effectiveSampleSize(samples)
      expect(ess).toBe(100) // constant → no variance → returns N
    })

    test('handles very short chains', () => {
      expect(effectiveSampleSize([1, 2])).toBe(2)
      expect(effectiveSampleSize([1, 2, 3])).toBe(3)
    })

    test('returns positive value', () => {
      // Alternating values (negative autocorrelation)
      const samples = []
      for (let i = 0; i < 200; i++) {
        samples.push(i % 2 === 0 ? 1 : -1)
      }
      const ess = effectiveSampleSize(samples)
      expect(ess).toBeGreaterThanOrEqual(1)
    })
  })

  describe('potentialScaleReduction', () => {
    test('R-hat ≈ 1 for converged chains', () => {
      // Multiple chains from the same distribution
      const n = 500
      const chains = []
      for (let c = 0; c < 4; c++) {
        const chain = []
        let seed = 10000 + c * 7919
        for (let i = 0; i < n; i++) {
          seed = (seed * 1103515245 + 12345) & 0x7fffffff
          // Standard normal approximation via Box-Muller
          const u1 = (seed / 0x7fffffff) || 0.001
          seed = (seed * 1103515245 + 12345) & 0x7fffffff
          const u2 = seed / 0x7fffffff
          chain.push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2))
        }
        chains.push(chain)
      }

      const rhat = potentialScaleReduction(chains)
      // Should be close to 1 for well-mixed chains
      expect(rhat).toBeGreaterThan(0.9)
      expect(rhat).toBeLessThan(1.2)
    })

    test('R-hat > 1 for non-converged chains', () => {
      // Chains with very different means
      const n = 100
      const chain1 = []; const chain2 = []
      let seed = 42
      for (let i = 0; i < n; i++) {
        seed = (seed * 1103515245 + 12345) & 0x7fffffff
        const u = seed / 0x7fffffff
        chain1.push(u + 0)    // centered at 0
        chain2.push(u + 10)   // centered at 10
      }

      const rhat = potentialScaleReduction([chain1, chain2])
      expect(rhat).toBeGreaterThan(1.5) // definitely not converged
    })

    test('R-hat ≈ 1 for identical chains', () => {
      // For identical chains B/n=0, finite-sample bias makes R-hat < 1
      // With m=2, n=10: R-hat = sqrt(0.9) ≈ 0.949
      const chain = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
      const rhat = potentialScaleReduction([chain, [...chain]])
      expect(rhat).toBeGreaterThan(0.9)
      expect(rhat).toBeLessThanOrEqual(1.0)
    })

    test('throws for single chain', () => {
      expect(() => potentialScaleReduction([[1, 2, 3]])).toThrow()
    })

    test('throws for chains with < 2 samples', () => {
      expect(() => potentialScaleReduction([[1], [2]])).toThrow()
    })

    test('works with 2 chains', () => {
      const n = 200
      const chains = [[], []]
      let seed = 99
      for (let i = 0; i < n; i++) {
        for (let c = 0; c < 2; c++) {
          seed = (seed * 1103515245 + 12345) & 0x7fffffff
          const u1 = (seed / 0x7fffffff) || 0.001
          seed = (seed * 1103515245 + 12345) & 0x7fffffff
          const u2 = seed / 0x7fffffff
          chains[c].push(
            Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
          )
        }
      }

      const rhat = potentialScaleReduction(chains)
      expect(typeof rhat).toBe('number')
      expect(isFinite(rhat)).toBe(true)
      expect(rhat).toBeGreaterThan(0)
    })
  })
})
