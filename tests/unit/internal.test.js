import * as tf from '@tensorflow/tfjs'
import {
  assertPositive, assertNonNegative, assertInRange, assertOneOf,
  commonDtype, isFloatDtype, isIntDtype,
  toTensor, broadcastShapes, broadcastShapesMultiple, shapeSize
} from '../../src/internal'

describe('assert-util', () => {
  describe('assertPositive', () => {
    test('passes for positive number', () => {
      assertPositive(1.0, 'x')
      assertPositive(0.001, 'x')
    })

    test('throws for zero', () => {
      expect(() => assertPositive(0, 'x')).toThrow('must be positive')
    })

    test('throws for negative', () => {
      expect(() => assertPositive(-1, 'x')).toThrow('must be positive')
    })

    test('throws for NaN', () => {
      expect(() => assertPositive(NaN, 'x')).toThrow('must be positive')
    })

    test('works with tensor', () => {
      const t = tf.tensor([1, 2, 3])
      assertPositive(t, 'x')
      t.dispose()
    })

    test('throws for tensor with non-positive', () => {
      const t = tf.tensor([1, -1, 3])
      expect(() => assertPositive(t, 'x')).toThrow('must be positive')
      t.dispose()
    })
  })

  describe('assertNonNegative', () => {
    test('passes for zero', () => {
      assertNonNegative(0, 'x')
    })

    test('throws for negative', () => {
      expect(() => assertNonNegative(-1, 'x')).toThrow('must be non-negative')
    })
  })

  describe('assertInRange', () => {
    test('passes for value in range', () => {
      assertInRange(0.5, 0, 1, 'x')
    })

    test('throws for value below range', () => {
      expect(() => assertInRange(-0.1, 0, 1, 'x')).toThrow('must be in')
    })

    test('throws for value above range', () => {
      expect(() => assertInRange(1.1, 0, 1, 'x')).toThrow('must be in')
    })

    test('works with tensor', () => {
      const t = tf.tensor([0.1, 0.5, 0.9])
      assertInRange(t, 0, 1, 'x')
      t.dispose()
    })
  })

  describe('assertOneOf', () => {
    test('passes when first provided', () => {
      assertOneOf(0.5, null, 'probs', 'logits')
    })

    test('passes when second provided', () => {
      assertOneOf(null, 0.5, 'probs', 'logits')
    })

    test('throws when both provided', () => {
      expect(() => assertOneOf(0.5, 0.5, 'probs', 'logits')).toThrow('Exactly one')
    })

    test('throws when neither provided', () => {
      expect(() => assertOneOf(null, null, 'probs', 'logits')).toThrow('Exactly one')
    })
  })
})

describe('dtype-util', () => {
  test('commonDtype prefers float32', () => {
    expect(commonDtype('float32', 'int32')).toBe('float32')
    expect(commonDtype('int32', 'float32')).toBe('float32')
  })

  test('commonDtype int32 + int32 = int32', () => {
    expect(commonDtype('int32', 'int32')).toBe('int32')
  })

  test('isFloatDtype', () => {
    expect(isFloatDtype('float32')).toBe(true)
    expect(isFloatDtype('int32')).toBe(false)
  })

  test('isIntDtype', () => {
    expect(isIntDtype('int32')).toBe(true)
    expect(isIntDtype('bool')).toBe(true)
    expect(isIntDtype('float32')).toBe(false)
  })
})

describe('tensor-util', () => {
  describe('toTensor', () => {
    test('converts number to scalar tensor', () => {
      const t = toTensor(5.0)
      expect(t.shape).toEqual([])
      expect(t.dataSync()[0]).toBeCloseTo(5.0)
      t.dispose()
    })

    test('converts array to tensor', () => {
      const t = toTensor([1, 2, 3])
      expect(t.shape).toEqual([3])
      t.dispose()
    })

    test('passes through existing tensor', () => {
      const orig = tf.tensor([1, 2])
      const t = toTensor(orig)
      expect(t).toBe(orig)
      t.dispose()
    })

    test('casts dtype when specified', () => {
      const t = toTensor(5, 'int32')
      expect(t.dtype).toBe('int32')
      t.dispose()
    })
  })

  describe('broadcastShapes', () => {
    test('same shapes', () => {
      expect(broadcastShapes([2, 3], [2, 3])).toEqual([2, 3])
    })

    test('scalar broadcast', () => {
      expect(broadcastShapes([], [2, 3])).toEqual([2, 3])
    })

    test('dimension 1 broadcast', () => {
      expect(broadcastShapes([1, 3], [2, 1])).toEqual([2, 3])
    })

    test('different ranks', () => {
      expect(broadcastShapes([3], [2, 3])).toEqual([2, 3])
    })

    test('throws for incompatible shapes', () => {
      expect(() => broadcastShapes([2], [3])).toThrow('not broadcast-compatible')
    })
  })

  describe('broadcastShapesMultiple', () => {
    test('handles multiple shapes', () => {
      expect(broadcastShapesMultiple([2, 1], [1, 3], [2, 3])).toEqual([2, 3])
    })

    test('handles empty input', () => {
      expect(broadcastShapesMultiple()).toEqual([])
    })
  })

  describe('shapeSize', () => {
    test('product of dimensions', () => {
      expect(shapeSize([2, 3, 4])).toBe(24)
    })

    test('single dimension', () => {
      expect(shapeSize([1000])).toBe(1000)
    })

    test('empty shape = 1', () => {
      expect(shapeSize([])).toBe(1)
    })

    test('null/undefined = 1', () => {
      expect(shapeSize(null)).toBe(1)
    })
  })
})
