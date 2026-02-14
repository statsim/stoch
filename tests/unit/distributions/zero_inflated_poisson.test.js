import * as tf from '@tensorflow/tfjs'
import { ZeroInflatedPoisson } from '../../../src/distributions/zero_inflated_poisson'
import { expectClose } from '../../helpers/tolerance'

describe('ZeroInflatedPoisson distribution', () => {
  describe('constructor', () => {
    test('basic params', () => {
      const d = new ZeroInflatedPoisson({ rate: 3, gate: 0.3 })
      expect(d.rate.dataSync()[0]).toBe(3)
      expect(d.gate.dataSync()[0]).toBeCloseTo(0.3)
      d.dispose()
    })
  })

  describe('sample', () => {
    test('samples are non-negative integers', () => {
      const d = new ZeroInflatedPoisson({ rate: 2, gate: 0.3 })
      const s = d.sample([100])
      const data = s.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThanOrEqual(0)
        expect(data[i] % 1).toBeCloseTo(0, 5) // integer
      }
      s.dispose()
      d.dispose()
    })

    test('more zeros than regular Poisson', () => {
      const d = new ZeroInflatedPoisson({ rate: 2, gate: 0.5 })
      const s = d.sample([1000])
      const data = s.dataSync()
      const zeros = data.filter(x => x === 0).length
      // With gate=0.5, rate=2: P(0) = 0.5 + 0.5*exp(-2) ≈ 0.568
      // Regular Poisson(2): P(0) = exp(-2) ≈ 0.135
      expect(zeros / 1000).toBeGreaterThan(0.4)
      s.dispose()
      d.dispose()
    })
  })

  describe('logProb', () => {
    test('logProb at 0 accounts for zero inflation', () => {
      const d = new ZeroInflatedPoisson({ rate: 2, gate: 0.3 })
      const lp = d.logProb(0)
      // P(0) = 0.3 + 0.7*exp(-2)
      const expected = Math.log(0.3 + 0.7 * Math.exp(-2))
      expectClose(lp.dataSync()[0], expected, { atol: 1e-3 })
      lp.dispose()
      d.dispose()
    })

    test('logProb at k>0 is (1-gate)*Poisson', () => {
      const d = new ZeroInflatedPoisson({ rate: 2, gate: 0.3 })
      const lp = d.logProb(3)
      // P(3) = 0.7 * 2^3 * exp(-2) / 3! = 0.7 * 8 * exp(-2) / 6
      const expected = Math.log(0.7) + 3 * Math.log(2) - 2 - Math.log(6)
      expectClose(lp.dataSync()[0], expected, { atol: 1e-3 })
      lp.dispose()
      d.dispose()
    })

    test('logProb P(0) > logProb P(k) for high gate', () => {
      const d = new ZeroInflatedPoisson({ rate: 1, gate: 0.8 })
      const lp0 = d.logProb(0).dataSync()[0]
      const lp1 = d.logProb(1).dataSync()[0]
      expect(lp0).toBeGreaterThan(lp1)
      d.dispose()
    })
  })

  describe('mean/variance', () => {
    test('mean = (1-gate)*rate', () => {
      const d = new ZeroInflatedPoisson({ rate: 3, gate: 0.4 })
      expectClose(d.mean().dataSync()[0], 0.6 * 3, { atol: 1e-4 })
      d.dispose()
    })

    test('variance = (1-gate)*rate*(1+gate*rate)', () => {
      const d = new ZeroInflatedPoisson({ rate: 2, gate: 0.3 })
      const expected = 0.7 * 2 * (1 + 0.3 * 2)
      expectClose(d.variance().dataSync()[0], expected, { atol: 1e-3 })
      d.dispose()
    })
  })
})
