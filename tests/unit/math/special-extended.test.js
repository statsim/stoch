import * as tf from '@tensorflow/tfjs'
import fs from 'fs'
import path from 'path'
import {
  logChoose, incompleteGamma, incompleteBeta,
  besselI0, besselI1, logBesselI0
} from '../../../src/math/special'
import { expectClose } from '../../helpers/tolerance'

const refPath = path.join(__dirname, '../../reference-data/math-special-reference.json')
let refData = null
try {
  refData = JSON.parse(fs.readFileSync(refPath, 'utf-8'))
} catch (e) {
  // Reference data not available — tests will skip
}

describe('math/special extended', () => {
  describe('logChoose', () => {
    if (!refData) {
      test.skip('reference data not available', () => {})
      return
    }

    for (const { n, k, expected } of refData.logChoose) {
      test(`logChoose(${n}, ${k}) ≈ ${expected.toFixed(4)}`, () => {
        const result = logChoose(n, k)
        expectClose(result, expected, { rtol: 1e-4, atol: 1e-6 })
      })
    }

    test('logChoose edge cases', () => {
      expect(logChoose(5, -1)).toBe(-Infinity)
      expect(logChoose(5, 6)).toBe(-Infinity)
      expect(logChoose(0, 0)).toBe(0)
      expect(logChoose(1, 0)).toBe(0)
      expect(logChoose(1, 1)).toBe(0)
    })
  })

  describe('incompleteGamma', () => {
    if (!refData) {
      test.skip('reference data not available', () => {})
      return
    }

    for (const { a, x, lower, upper } of refData.incompleteGamma) {
      test(`P(${a}, ${x}) ≈ ${lower.toFixed(6)}`, () => {
        const result = incompleteGamma(a, x)
        // float32 logGamma limits precision for extreme params
        expectClose(result.lower, lower, { rtol: 1e-4, atol: 1e-6 })
        expectClose(result.upper, upper, { rtol: 1e-4, atol: 1e-6 })
        expectClose(result.lower + result.upper, 1.0, { atol: 1e-8 })
      })
    }

    test('incompleteGamma(a, 0) = { lower: 0, upper: 1 }', () => {
      const result = incompleteGamma(1, 0)
      expect(result.lower).toBe(0)
      expect(result.upper).toBe(1)
    })

    test('throws on invalid inputs', () => {
      expect(() => incompleteGamma(1, -1)).toThrow()
      expect(() => incompleteGamma(0, 1)).toThrow()
    })
  })

  describe('incompleteBeta', () => {
    if (!refData) {
      test.skip('reference data not available', () => {})
      return
    }

    for (const { a, b, x, expected } of refData.incompleteBeta) {
      test(`I_${x}(${a}, ${b}) ≈ ${expected.toFixed(6)}`, () => {
        const result = incompleteBeta(a, b, x)
        // float32 logBeta limits precision for extreme params
        expectClose(result, expected, { rtol: 1e-4, atol: 1e-5 })
      })
    }

    test('boundary values', () => {
      expect(incompleteBeta(1, 1, 0)).toBe(0)
      expect(incompleteBeta(1, 1, 1)).toBe(1)
    })

    test('symmetry: I_0.5(a, a) = 0.5', () => {
      expectClose(incompleteBeta(2, 2, 0.5), 0.5, { atol: 1e-10 })
      expectClose(incompleteBeta(5, 5, 0.5), 0.5, { atol: 1e-10 })
    })

    test('throws on invalid inputs', () => {
      expect(() => incompleteBeta(1, 1, -0.1)).toThrow()
      expect(() => incompleteBeta(1, 1, 1.1)).toThrow()
    })
  })

  describe('besselI0', () => {
    if (!refData) {
      test.skip('reference data not available', () => {})
      return
    }

    for (const { x, i0 } of refData.besselI) {
      test(`I₀(${x}) ≈ ${i0.toFixed(6)}`, () => {
        const result = besselI0(x)
        expectClose(result, i0, { rtol: 1e-5, atol: 1e-7 })
      })
    }

    test('I₀(0) = 1', () => {
      expect(besselI0(0)).toBe(1)
    })
  })

  describe('besselI1', () => {
    if (!refData) {
      test.skip('reference data not available', () => {})
      return
    }

    for (const { x, i1 } of refData.besselI) {
      test(`I₁(${x}) ≈ ${i1.toFixed(6)}`, () => {
        const result = besselI1(x)
        expectClose(result, i1, { rtol: 1e-5, atol: 1e-7 })
      })
    }

    test('I₁(0) = 0', () => {
      expect(besselI1(0)).toBe(0)
    })
  })

  describe('logBesselI0', () => {
    if (!refData) {
      test.skip('reference data not available', () => {})
      return
    }

    for (const { x, logI0 } of refData.besselI) {
      test(`logI₀(${x}) ≈ ${logI0.toFixed(6)}`, () => {
        const result = logBesselI0(x)
        expectClose(result, logI0, { rtol: 1e-4, atol: 1e-6 })
      })
    }

    test('consistency: logBesselI0(x) ≈ log(besselI0(x))', () => {
      const testX = [0.1, 1.0, 5.0, 10.0]
      for (const x of testX) {
        expectClose(logBesselI0(x), Math.log(besselI0(x)), { rtol: 1e-6, atol: 1e-8 })
      }
    })
  })
})
