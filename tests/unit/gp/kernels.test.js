import * as tf from '@tensorflow/tfjs'
import {
  SquaredExponential, Matern, Linear, Periodic, White,
  Add, Product, Scale
} from '../../../src/gp/kernels'

describe('GP Kernels', () => {
  const x1 = tf.tensor2d([[0], [1], [2]])       // [3, 1]
  const x2 = tf.tensor2d([[0], [1], [2], [3]])   // [4, 1]

  describe('SquaredExponential (RBF)', () => {
    test('kernel matrix is symmetric for self-covariance', () => {
      const k = new SquaredExponential()
      const K = k.matrix(x1, x1)
      expect(K.shape).toEqual([3, 3])
      const data = K.dataSync()
      // K[i,j] should equal K[j,i]
      expect(data[1]).toBeCloseTo(data[3], 5)  // K[0,1] == K[1,0]
      expect(data[2]).toBeCloseTo(data[6], 5)  // K[0,2] == K[2,0]
      K.dispose()
    })

    test('diagonal is amplitude²', () => {
      const k = new SquaredExponential({ amplitude: 2 })
      const K = k.matrix(x1, x1)
      const data = K.dataSync()
      expect(data[0]).toBeCloseTo(4, 4) // 2² = 4
      expect(data[4]).toBeCloseTo(4, 4)
      expect(data[8]).toBeCloseTo(4, 4)
      K.dispose()
    })

    test('length scale controls decay', () => {
      const kShort = new SquaredExponential({ lengthScale: 0.5 })
      const kLong = new SquaredExponential({ lengthScale: 2 })
      const Ks = kShort.matrix(x1, x1)
      const Kl = kLong.matrix(x1, x1)
      // K_short[0,1] < K_long[0,1] (shorter length scale = faster decay)
      expect(Ks.dataSync()[1]).toBeLessThan(Kl.dataSync()[1])
      Ks.dispose(); Kl.dispose()
    })

    test('matrix shape for cross-covariance', () => {
      const k = new SquaredExponential()
      const K = k.matrix(x1, x2)
      expect(K.shape).toEqual([3, 4])
      K.dispose()
    })

    test('apply returns pointwise values', () => {
      const k = new SquaredExponential()
      const vals = k.apply(x1, x1)
      expect(vals.shape).toEqual([3])
      // k(x, x) = 1 for amplitude=1
      const data = vals.dataSync()
      expect(data[0]).toBeCloseTo(1, 5)
      expect(data[1]).toBeCloseTo(1, 5)
      vals.dispose()
    })
  })

  describe('Matern', () => {
    test('Matern 0.5 (exponential)', () => {
      const k = new Matern({ nu: 0.5 })
      const K = k.matrix(x1, x1)
      expect(K.shape).toEqual([3, 3])
      expect(K.dataSync()[0]).toBeCloseTo(1, 3) // k(0,0) ≈ 1
      K.dispose()
    })

    test('Matern 1.5', () => {
      const k = new Matern({ nu: 1.5 })
      const K = k.matrix(x1, x1)
      expect(K.dataSync()[0]).toBeCloseTo(1, 3)
      K.dispose()
    })

    test('Matern 2.5', () => {
      const k = new Matern({ nu: 2.5 })
      const K = k.matrix(x1, x1)
      expect(K.dataSync()[0]).toBeCloseTo(1, 3)
      K.dispose()
    })

    test('smoother kernels decay slower', () => {
      const k05 = new Matern({ nu: 0.5 })
      const k25 = new Matern({ nu: 2.5 })
      // At distance 1: Matern 2.5 should have higher covariance than 0.5
      const K05 = k05.matrix(x1, x1)
      const K25 = k25.matrix(x1, x1)
      expect(K25.dataSync()[1]).toBeGreaterThan(K05.dataSync()[1])
      K05.dispose(); K25.dispose()
    })

    test('throws for invalid nu', () => {
      expect(() => new Matern({ nu: 1.0 })).toThrow()
    })
  })

  describe('Linear', () => {
    test('kernel matrix', () => {
      const k = new Linear()
      const K = k.matrix(x1, x1)
      expect(K.shape).toEqual([3, 3])
      // K[i,j] = x1[i] * x1[j] (bias=0, variance=1)
      expect(K.dataSync()[0]).toBeCloseTo(0, 5) // 0*0
      expect(K.dataSync()[4]).toBeCloseTo(1, 5) // 1*1
      expect(K.dataSync()[8]).toBeCloseTo(4, 5) // 2*2
      K.dispose()
    })

    test('bias shifts center', () => {
      const k = new Linear({ bias: 1 })
      const K = k.matrix(x1, x1)
      // K[i,j] = (x[i]-1)*(x[j]-1)
      expect(K.dataSync()[0]).toBeCloseTo(1, 5)  // (-1)*(-1) = 1
      expect(K.dataSync()[4]).toBeCloseTo(0, 5)  // 0*0 = 0
      expect(K.dataSync()[8]).toBeCloseTo(1, 5)  // 1*1 = 1
      K.dispose()
    })
  })

  describe('Periodic', () => {
    test('kernel is periodic', () => {
      const k = new Periodic({ period: 2 })
      const xa = tf.tensor2d([[0], [2], [4]]) // period apart
      const K = k.matrix(xa, xa)
      // k(0, 2) should be close to k(0, 0) since period=2
      expect(K.dataSync()[1]).toBeCloseTo(K.dataSync()[0], 2)
      K.dispose(); xa.dispose()
    })

    test('diagonal is amplitude²', () => {
      const k = new Periodic({ amplitude: 3 })
      const K = k.matrix(x1, x1)
      expect(K.dataSync()[0]).toBeCloseTo(9, 3)
      K.dispose()
    })
  })

  describe('White', () => {
    test('identity matrix for self-covariance', () => {
      const k = new White({ variance: 2 })
      const K = k.matrix(x1, x1)
      expect(K.shape).toEqual([3, 3])
      const data = K.dataSync()
      expect(data[0]).toBeCloseTo(2, 5)
      expect(data[4]).toBeCloseTo(2, 5)
      expect(data[1]).toBeCloseTo(0, 5)
      K.dispose()
    })

    test('zeros for cross-covariance', () => {
      const k = new White()
      const K = k.matrix(x1, x2)
      const data = K.dataSync()
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeCloseTo(0, 5)
      }
      K.dispose()
    })
  })

  describe('Combinators', () => {
    test('Add combines two kernels', () => {
      const k = new Add(
        new SquaredExponential(),
        new White({ variance: 0.1 })
      )
      const K = k.matrix(x1, x1)
      expect(K.shape).toEqual([3, 3])
      // Diagonal should be 1 + 0.1 = 1.1
      expect(K.dataSync()[0]).toBeCloseTo(1.1, 3)
      K.dispose()
    })

    test('Product multiplies two kernels', () => {
      const k = new Product(
        new SquaredExponential(),
        new Linear()
      )
      const K = k.matrix(x1, x1)
      expect(K.shape).toEqual([3, 3])
      // K[0,0] = SE(0,0) * Lin(0,0) = 1 * 0 = 0
      expect(K.dataSync()[0]).toBeCloseTo(0, 5)
      // K[1,1] = SE(1,1) * Lin(1,1) = 1 * 1 = 1
      expect(K.dataSync()[4]).toBeCloseTo(1, 3)
      K.dispose()
    })

    test('Scale multiplies by constant', () => {
      const k = new Scale(new SquaredExponential(), 3)
      const K = k.matrix(x1, x1)
      expect(K.dataSync()[0]).toBeCloseTo(3, 3)
      K.dispose()
    })
  })
})
