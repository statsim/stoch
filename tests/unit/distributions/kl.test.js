import * as tf from '@tensorflow/tfjs'
import * as fs from 'fs'
import * as path from 'path'
import { Normal } from '../../../src/distributions/normal'
import { Bernoulli } from '../../../src/distributions/bernoulli'
import { Gamma } from '../../../src/distributions/gamma'
import { Beta } from '../../../src/distributions/beta'
import { Exponential } from '../../../src/distributions/exponential'
import { Dirichlet } from '../../../src/distributions/dirichlet'
import { Categorical } from '../../../src/distributions/categorical'
import { Laplace } from '../../../src/distributions/laplace'
import { klDivergence } from '../../../src/distributions/kullback_leibler'
import { expectClose } from '../../helpers/tolerance'

describe('KL divergence', () => {
  describe('Normal-Normal', () => {
    test('KL(N(0,1) || N(0,1)) = 0', () => {
      const p = new Normal({ loc: 0, scale: 1 })
      const q = new Normal({ loc: 0, scale: 1 })
      const kl = klDivergence(p, q)
      expectClose(kl.dataSync()[0], 0, { atol: 1e-5 })
      kl.dispose()
      p.dispose()
      q.dispose()
    })

    test('KL(N(0,1) || N(1,2))', () => {
      const p = new Normal({ loc: 0, scale: 1 })
      const q = new Normal({ loc: 1, scale: 2 })
      const kl = klDivergence(p, q)
      // KL = log(2/1) + (1 + 1) / (2*4) - 0.5 = log(2) + 0.25 - 0.5
      const expected = Math.log(2) + 0.25 - 0.5
      expectClose(kl.dataSync()[0], expected, { rtol: 1e-3 })
      kl.dispose()
      p.dispose()
      q.dispose()
    })

    test('KL is non-negative', () => {
      const p = new Normal({ loc: -1, scale: 0.5 })
      const q = new Normal({ loc: 2, scale: 3 })
      const kl = klDivergence(p, q)
      expect(kl.dataSync()[0]).toBeGreaterThanOrEqual(0)
      kl.dispose()
      p.dispose()
      q.dispose()
    })

    test('matches reference data', () => {
      const refPath = path.join(__dirname, '../../reference-data/normal.json')
      if (!fs.existsSync(refPath)) return
      const refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))

      for (const klCase of refData.kl_divergence) {
        const p = new Normal({ loc: klCase.p.loc, scale: klCase.p.scale })
        const q = new Normal({ loc: klCase.q.loc, scale: klCase.q.scale })
        const kl = klDivergence(p, q)
        expectClose(kl.dataSync()[0], klCase.expected, { rtol: 1e-3, atol: 1e-4 })
        kl.dispose()
        p.dispose()
        q.dispose()
      }
    })
  })

  describe('Bernoulli-Bernoulli', () => {
    test('KL(Ber(0.5) || Ber(0.5)) = 0', () => {
      const p = new Bernoulli({ probs: 0.5 })
      const q = new Bernoulli({ probs: 0.5 })
      const kl = klDivergence(p, q)
      expectClose(kl.dataSync()[0], 0, { atol: 1e-5 })
      kl.dispose()
      p.dispose()
      q.dispose()
    })

    test('KL(Ber(0.3) || Ber(0.7))', () => {
      const p = new Bernoulli({ probs: 0.3 })
      const q = new Bernoulli({ probs: 0.7 })
      const kl = klDivergence(p, q)
      // 0.3*log(0.3/0.7) + 0.7*log(0.7/0.3)
      const expected = 0.3 * Math.log(0.3 / 0.7) + 0.7 * Math.log(0.7 / 0.3)
      expectClose(kl.dataSync()[0], expected, { rtol: 1e-3 })
      kl.dispose()
      p.dispose()
      q.dispose()
    })
  })

  describe('Gamma-Gamma', () => {
    test('KL(Gamma(a,b) || Gamma(a,b)) = 0', () => {
      const p = new Gamma({ concentration: 2, rate: 1 })
      const q = new Gamma({ concentration: 2, rate: 1 })
      const kl = klDivergence(p, q)
      expectClose(kl.dataSync()[0], 0, { atol: 1e-4 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL(Gamma(2,1) || Gamma(3,2))', () => {
      const p = new Gamma({ concentration: 2, rate: 1 })
      const q = new Gamma({ concentration: 3, rate: 2 })
      const kl = klDivergence(p, q)
      // (a_p - a_q)*psi(a_p) - lgamma(a_p) + lgamma(a_q) + a_q*ln(b_p/b_q) + a_p*(b_q/b_p - 1)
      // psi(2) ≈ 0.4227843, lgamma(2) = 0, lgamma(3) = ln(2) ≈ 0.6931
      const psi2 = 0.42278433509846714
      const expected = (2 - 3) * psi2 - 0 + Math.log(2) + 3 * Math.log(1 / 2) + 2 * (2 / 1 - 1)
      expectClose(kl.dataSync()[0], expected, { rtol: 1e-3, atol: 1e-3 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL is non-negative', () => {
      const p = new Gamma({ concentration: 1, rate: 2 })
      const q = new Gamma({ concentration: 3, rate: 0.5 })
      const kl = klDivergence(p, q)
      expect(kl.dataSync()[0]).toBeGreaterThanOrEqual(-1e-5)
      kl.dispose(); p.dispose(); q.dispose()
    })
  })

  describe('Beta-Beta', () => {
    test('KL(Beta(a,b) || Beta(a,b)) = 0', () => {
      const p = new Beta({ concentration1: 2, concentration0: 3 })
      const q = new Beta({ concentration1: 2, concentration0: 3 })
      const kl = klDivergence(p, q)
      expectClose(kl.dataSync()[0], 0, { atol: 1e-4 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL(Beta(2,3) || Beta(1,1))', () => {
      const p = new Beta({ concentration1: 2, concentration0: 3 })
      const q = new Beta({ concentration1: 1, concentration0: 1 })
      const kl = klDivergence(p, q)
      // KL to uniform Beta(1,1) should be non-negative
      expect(kl.dataSync()[0]).toBeGreaterThanOrEqual(-1e-5)
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL is non-negative', () => {
      const p = new Beta({ concentration1: 0.5, concentration0: 0.5 })
      const q = new Beta({ concentration1: 2, concentration0: 2 })
      const kl = klDivergence(p, q)
      expect(kl.dataSync()[0]).toBeGreaterThanOrEqual(-1e-5)
      kl.dispose(); p.dispose(); q.dispose()
    })
  })

  describe('Exponential-Exponential', () => {
    test('KL(Exp(r) || Exp(r)) = 0', () => {
      const p = new Exponential({ rate: 2 })
      const q = new Exponential({ rate: 2 })
      const kl = klDivergence(p, q)
      expectClose(kl.dataSync()[0], 0, { atol: 1e-5 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL(Exp(1) || Exp(2))', () => {
      const p = new Exponential({ rate: 1 })
      const q = new Exponential({ rate: 2 })
      const kl = klDivergence(p, q)
      // log(2/1) + 1/2 - 1 = log(2) - 0.5
      const expected = Math.log(2) + 0.5 - 1
      expectClose(kl.dataSync()[0], expected, { rtol: 1e-3 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL is non-negative', () => {
      const p = new Exponential({ rate: 0.5 })
      const q = new Exponential({ rate: 3 })
      const kl = klDivergence(p, q)
      expect(kl.dataSync()[0]).toBeGreaterThanOrEqual(-1e-5)
      kl.dispose(); p.dispose(); q.dispose()
    })
  })

  describe('Dirichlet-Dirichlet', () => {
    test('KL(Dir(a) || Dir(a)) = 0', () => {
      const p = new Dirichlet({ concentration: [1, 2, 3] })
      const q = new Dirichlet({ concentration: [1, 2, 3] })
      const kl = klDivergence(p, q)
      expectClose(kl.dataSync()[0], 0, { atol: 1e-4 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL(Dir(1,1,1) || Dir(2,2,2))', () => {
      const p = new Dirichlet({ concentration: [1, 1, 1] })
      const q = new Dirichlet({ concentration: [2, 2, 2] })
      const kl = klDivergence(p, q)
      // Should be non-negative
      expect(kl.dataSync()[0]).toBeGreaterThanOrEqual(-1e-5)
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL is non-negative', () => {
      const p = new Dirichlet({ concentration: [0.5, 1, 2] })
      const q = new Dirichlet({ concentration: [3, 0.5, 1] })
      const kl = klDivergence(p, q)
      expect(kl.dataSync()[0]).toBeGreaterThanOrEqual(-1e-5)
      kl.dispose(); p.dispose(); q.dispose()
    })
  })

  describe('Categorical-Categorical', () => {
    test('KL(Cat(p) || Cat(p)) = 0', () => {
      const p = new Categorical({ probs: [0.2, 0.3, 0.5] })
      const q = new Categorical({ probs: [0.2, 0.3, 0.5] })
      const kl = klDivergence(p, q)
      expectClose(kl.dataSync()[0], 0, { atol: 1e-5 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL(Cat([0.5,0.5]) || Cat([0.25,0.75]))', () => {
      const p = new Categorical({ probs: [0.5, 0.5] })
      const q = new Categorical({ probs: [0.25, 0.75] })
      const kl = klDivergence(p, q)
      // 0.5*log(0.5/0.25) + 0.5*log(0.5/0.75) = 0.5*log(2) + 0.5*log(2/3)
      const expected = 0.5 * Math.log(2) + 0.5 * Math.log(2 / 3)
      expectClose(kl.dataSync()[0], expected, { rtol: 1e-3 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL is non-negative', () => {
      const p = new Categorical({ probs: [0.1, 0.2, 0.7] })
      const q = new Categorical({ probs: [0.4, 0.4, 0.2] })
      const kl = klDivergence(p, q)
      expect(kl.dataSync()[0]).toBeGreaterThanOrEqual(-1e-5)
      kl.dispose(); p.dispose(); q.dispose()
    })
  })

  describe('Laplace-Laplace', () => {
    test('KL(Lap(l,s) || Lap(l,s)) = 0', () => {
      const p = new Laplace({ loc: 0, scale: 1 })
      const q = new Laplace({ loc: 0, scale: 1 })
      const kl = klDivergence(p, q)
      expectClose(kl.dataSync()[0], 0, { atol: 1e-5 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL(Lap(0,1) || Lap(1,2))', () => {
      const p = new Laplace({ loc: 0, scale: 1 })
      const q = new Laplace({ loc: 1, scale: 2 })
      const kl = klDivergence(p, q)
      // log(2/1) + |0-1|/2 + (1/2)*exp(-|0-1|/1) - 1
      // = log(2) + 0.5 + 0.5*exp(-1) - 1
      const expected = Math.log(2) + 0.5 + 0.5 * Math.exp(-1) - 1
      expectClose(kl.dataSync()[0], expected, { rtol: 1e-3 })
      kl.dispose(); p.dispose(); q.dispose()
    })

    test('KL is non-negative', () => {
      const p = new Laplace({ loc: -2, scale: 0.5 })
      const q = new Laplace({ loc: 3, scale: 2 })
      const kl = klDivergence(p, q)
      expect(kl.dataSync()[0]).toBeGreaterThanOrEqual(-1e-5)
      kl.dispose(); p.dispose(); q.dispose()
    })
  })

  test('throws for unregistered pair', () => {
    const p = new Normal()
    const q = new Bernoulli({ probs: 0.5 })
    expect(() => klDivergence(p, q)).toThrow('No KL divergence registered')
    p.dispose()
    q.dispose()
  })
})
