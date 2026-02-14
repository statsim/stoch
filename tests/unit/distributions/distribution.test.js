import * as tf from '@tensorflow/tfjs'
import { Distribution } from '../../../src/distributions/distribution'

// Minimal concrete subclass for testing the base class
class TestDist extends Distribution {
  constructor({ a = 0, b = 1, validateArgs } = {}) {
    super({ dtype: 'float32', validateArgs, name: 'TestDist' })
    this._a = this._addParameter('a', a)
    this._b = this._addParameter('b', b)
  }

  _sampleN(n) {
    return tf.randomUniform([n, ...this.batchShape])
  }

  _logProb(value) {
    return tf.zerosLike(value)
  }

  _cdf(value) {
    return tf.onesLike(value)
  }

  _entropy() {
    return tf.scalar(1.0)
  }

  _mean() {
    return this._a
  }

  _variance() {
    return tf.scalar(0.5)
  }

  _mode() {
    return this._a
  }
}

describe('Distribution base class', () => {
  test('constructor sets properties', () => {
    const d = new TestDist()
    expect(d.dtype).toBe('float32')
    expect(d.name).toBe('TestDist')
    d.dispose()
  })

  test('parameters are stored', () => {
    const d = new TestDist({ a: 5, b: 10 })
    expect(d.parameters.a).toBe(5)
    expect(d.parameters.b).toBe(10)
    d.dispose()
  })

  test('batchShape from scalar params', () => {
    const d = new TestDist({ a: 0, b: 1 })
    expect(d.batchShape).toEqual([])
    d.dispose()
  })

  test('batchShape from array params', () => {
    const d = new TestDist({ a: [1, 2, 3], b: 1 })
    expect(d.batchShape).toEqual([3])
    d.dispose()
  })

  test('batchShape from broadcast', () => {
    const d = new TestDist({ a: [[1, 2], [3, 4]], b: [1, 2] })
    expect(d.batchShape).toEqual([2, 2])
    d.dispose()
  })

  test('eventShape default is []', () => {
    const d = new TestDist()
    expect(d.eventShape).toEqual([])
    d.dispose()
  })

  describe('public methods wrap in tf.tidy', () => {
    test('logProb returns tensor and cleans up intermediates', () => {
      const d = new TestDist()
      const before = tf.memory().numTensors
      const result = d.logProb(0.5)
      expect(result instanceof tf.Tensor).toBe(true)
      result.dispose()
      d.dispose()
    })

    test('prob defaults to exp(logProb)', () => {
      const d = new TestDist()
      const result = d.prob(0.5)
      // logProb returns 0, so prob should be exp(0) = 1
      expect(result.dataSync()[0]).toBeCloseTo(1.0, 5)
      result.dispose()
      d.dispose()
    })

    test('cdf returns tensor', () => {
      const d = new TestDist()
      const result = d.cdf(0.5)
      expect(result.dataSync()[0]).toBeCloseTo(1.0, 5)
      result.dispose()
      d.dispose()
    })

    test('logCdf defaults to log(cdf)', () => {
      const d = new TestDist()
      const result = d.logCdf(0.5)
      expect(result.dataSync()[0]).toBeCloseTo(0.0, 5)
      result.dispose()
      d.dispose()
    })

    test('entropy returns tensor', () => {
      const d = new TestDist()
      const result = d.entropy()
      expect(result.dataSync()[0]).toBeCloseTo(1.0, 5)
      result.dispose()
      d.dispose()
    })

    test('mean returns tensor', () => {
      const d = new TestDist({ a: 5 })
      const result = d.mean()
      expect(result.dataSync()[0]).toBeCloseTo(5.0, 5)
      result.dispose()
      d.dispose()
    })

    test('variance returns tensor', () => {
      const d = new TestDist()
      const result = d.variance()
      expect(result.dataSync()[0]).toBeCloseTo(0.5, 5)
      result.dispose()
      d.dispose()
    })

    test('stddev defaults to sqrt(variance)', () => {
      const d = new TestDist()
      const result = d.stddev()
      expect(result.dataSync()[0]).toBeCloseTo(Math.sqrt(0.5), 4)
      result.dispose()
      d.dispose()
    })

    test('mode returns tensor', () => {
      const d = new TestDist({ a: 3 })
      const result = d.mode()
      expect(result.dataSync()[0]).toBeCloseTo(3.0, 5)
      result.dispose()
      d.dispose()
    })
  })

  test('sample returns correct shape', () => {
    const d = new TestDist()
    const s1 = d.sample()
    expect(s1.shape).toEqual([1])
    s1.dispose()

    const s2 = d.sample([5])
    expect(s2.shape).toEqual([5])
    s2.dispose()

    const s3 = d.sample(10)
    expect(s3.shape).toEqual([10])
    s3.dispose()

    d.dispose()
  })

  test('sample with batched distribution', () => {
    const d = new TestDist({ a: [1, 2, 3], b: 1 })
    const s = d.sample([5])
    expect(s.shape).toEqual([5, 3])
    s.dispose()
    d.dispose()
  })

  test('dispose frees parameter tensors', () => {
    const before = tf.memory().numTensors
    const d = new TestDist({ a: 5, b: 10 })
    const afterCreate = tf.memory().numTensors
    expect(afterCreate).toBeGreaterThan(before)
    d.dispose()
    const afterDispose = tf.memory().numTensors
    expect(afterDispose).toBe(before)
  })

  test('abstract methods throw', () => {
    const d = new Distribution({ name: 'Abstract' })
    expect(() => d.sample()).toThrow('_sampleN not implemented')
    expect(() => d.logProb(0)).toThrow('_logProb not implemented')
    expect(() => d.cdf(0)).toThrow('_cdf not implemented')
    expect(() => d.entropy()).toThrow('_entropy not implemented')
    expect(() => d.mean()).toThrow('_mean not implemented')
    expect(() => d.variance()).toThrow('_variance not implemented')
    expect(() => d.mode()).toThrow('_mode not implemented')
    d.dispose()
  })

  test('toString', () => {
    const d = new TestDist({ a: 0, b: 1 })
    expect(d.toString()).toContain('TestDist')
    d.dispose()
  })
})
