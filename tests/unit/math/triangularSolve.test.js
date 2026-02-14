import * as tf from '@tensorflow/tfjs'
import { triangularSolve } from '../../../src/math/triangularSolve'
import { expectTensorClose } from '../../helpers/tolerance'

describe('math/triangularSolve', () => {
  describe('lower triangular, no adjoint', () => {
    test('2x2 lower solve', () => {
      // L = [[2, 0], [1, 3]], B = [[4], [7]]
      // L·X = B => X = [[2], [5/3]]
      const L = tf.tensor2d([[2, 0], [1, 3]])
      const B = tf.tensor2d([[4], [7]])
      const X = triangularSolve(L, B)
      expect(X.shape).toEqual([2, 1])
      expectTensorClose(X, tf.tensor2d([[2], [5 / 3]]), { rtol: 1e-5, atol: 1e-6 })
      X.dispose(); L.dispose(); B.dispose()
    })

    test('3x3 lower solve', () => {
      // L = [[1, 0, 0], [2, 3, 0], [4, 5, 6]]
      // B = [[1], [8], [32]]
      // Row 0: x0 = 1/1 = 1
      // Row 1: x1 = (8 - 2*1)/3 = 2
      // Row 2: x2 = (32 - 4*1 - 5*2)/6 = 3
      const L = tf.tensor2d([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
      const B = tf.tensor2d([[1], [8], [32]])
      const X = triangularSolve(L, B)
      expectTensorClose(X, tf.tensor2d([[1], [2], [3]]), { rtol: 1e-5, atol: 1e-6 })
      X.dispose(); L.dispose(); B.dispose()
    })

    test('multiple RHS columns [3, 2]', () => {
      const L = tf.tensor2d([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
      // B has 2 columns
      const B = tf.tensor2d([[1, 6], [8, 21], [32, 74]])
      const X = triangularSolve(L, B)
      expect(X.shape).toEqual([3, 2])
      // Verify L·X ≈ B
      const reconstructed = tf.matMul(L, X)
      expectTensorClose(reconstructed, B, { rtol: 1e-4, atol: 1e-5 })
      X.dispose(); L.dispose(); B.dispose(); reconstructed.dispose()
    })
  })

  describe('adjoint (Lᵀ solve)', () => {
    test('2x2 Lᵀ solve', () => {
      // L = [[2, 0], [1, 3]], Lᵀ = [[2, 1], [0, 3]]
      // Lᵀ·X = B => X = backsolve
      // B = [[5], [6]]
      // i=1: x1 = 6/3 = 2
      // i=0: x0 = (5 - 1*2)/2 = 1.5
      const L = tf.tensor2d([[2, 0], [1, 3]])
      const B = tf.tensor2d([[5], [6]])
      const X = triangularSolve(L, B, { lower: true, adjoint: true })
      expectTensorClose(X, tf.tensor2d([[1.5], [2]]), { rtol: 1e-5, atol: 1e-6 })
      // Verify: Lᵀ · X = B
      const Lt = tf.transpose(L)
      const reconstructed = tf.matMul(Lt, X)
      expectTensorClose(reconstructed, B, { rtol: 1e-5, atol: 1e-6 })
      X.dispose(); L.dispose(); B.dispose(); Lt.dispose(); reconstructed.dispose()
    })

    test('3x3 Lᵀ solve', () => {
      const L = tf.tensor2d([[1, 0, 0], [2, 3, 0], [4, 5, 6]])
      const B = tf.tensor2d([[10], [20], [18]])
      const X = triangularSolve(L, B, { lower: true, adjoint: true })
      // Verify: Lᵀ · X = B
      const Lt = tf.transpose(L)
      const reconstructed = tf.matMul(Lt, X)
      expectTensorClose(reconstructed, B, { rtol: 1e-4, atol: 1e-5 })
      X.dispose(); L.dispose(); B.dispose(); Lt.dispose(); reconstructed.dispose()
    })
  })

  describe('batch support', () => {
    test('batched [2, 3, 3] solve', () => {
      const L = tf.tensor([
        [[1, 0, 0], [2, 3, 0], [4, 5, 6]],
        [[2, 0, 0], [1, 4, 0], [3, 2, 5]]
      ])
      const B = tf.tensor([
        [[1], [8], [32]],
        [[2], [9], [23]]
      ])
      const X = triangularSolve(L, B)
      expect(X.shape).toEqual([2, 3, 1])
      // Verify L·X ≈ B for each batch
      const reconstructed = tf.matMul(L, X)
      expectTensorClose(reconstructed, B, { rtol: 1e-4, atol: 1e-5 })
      X.dispose(); L.dispose(); B.dispose(); reconstructed.dispose()
    })
  })

  describe('identity matrix', () => {
    test('I · X = B => X = B', () => {
      const I = tf.eye(3)
      const B = tf.tensor2d([[1], [2], [3]])
      const X = triangularSolve(I, B)
      expectTensorClose(X, B, { atol: 1e-6 })
      X.dispose(); I.dispose(); B.dispose()
    })
  })

  describe('JS array input', () => {
    test('accepts plain JS arrays', () => {
      const X = triangularSolve([[2, 0], [1, 3]], [[4], [7]])
      expect(X.shape).toEqual([2, 1])
      expectTensorClose(X, tf.tensor2d([[2], [5 / 3]]), { rtol: 1e-5, atol: 1e-6 })
      X.dispose()
    })
  })

  describe('vector rhs', () => {
    test('1D rhs is treated as column vector', () => {
      const L = tf.tensor2d([[2, 0], [1, 3]])
      const b = tf.tensor1d([4, 7])
      const x = triangularSolve(L, b)
      expect(x.shape).toEqual([2])
      const data = x.dataSync()
      expect(data[0]).toBeCloseTo(2, 5)
      expect(data[1]).toBeCloseTo(5 / 3, 4)
      x.dispose(); L.dispose(); b.dispose()
    })
  })

  describe('error cases', () => {
    test('throws on singular matrix (zero diagonal)', () => {
      const L = tf.tensor2d([[1, 0], [1, 0]])
      const B = tf.tensor2d([[1], [1]])
      expect(() => triangularSolve(L, B)).toThrow('singular matrix')
      L.dispose(); B.dispose()
    })

    test('throws on non-square matrix', () => {
      const L = tf.tensor2d([[1, 2, 3], [4, 5, 6]])
      const B = tf.tensor2d([[1], [1]])
      expect(() => triangularSolve(L, B)).toThrow('square')
      L.dispose(); B.dispose()
    })

    test('throws on incompatible rhs shape', () => {
      const L = tf.tensor2d([[1, 0], [1, 1]])
      const B = tf.tensor2d([[1], [2], [3]])
      expect(() => triangularSolve(L, B)).toThrow('incompatible')
      L.dispose(); B.dispose()
    })

    test('throws on 1D matrix', () => {
      expect(() => triangularSolve(tf.tensor1d([1, 2]), tf.tensor1d([1, 2]))).toThrow('2D')
    })
  })

  describe('memory management', () => {
    test('does not leak tensors', () => {
      const L = tf.tensor2d([[2, 0], [1, 3]])
      const B = tf.tensor2d([[4], [7]])
      const before = tf.memory().numTensors
      const X = triangularSolve(L, B)
      X.dispose()
      expect(tf.memory().numTensors).toBe(before)
      L.dispose(); B.dispose()
    })
  })
})
