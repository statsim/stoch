import * as tf from '@tensorflow/tfjs'
import { Identity } from '../../../src/bijectors/identity'
import { Exp } from '../../../src/bijectors/exp'
import { Log } from '../../../src/bijectors/log'
import { Softplus } from '../../../src/bijectors/softplus'
import { Sigmoid } from '../../../src/bijectors/sigmoid'

// Helper: check forward-inverse roundtrip
function checkRoundtrip(bijector, points, tol = 1e-4) {
  const x = tf.tensor(points)
  const y = bijector.forward(x)
  const xBack = bijector.inverse(y)
  const xData = x.dataSync()
  const xBackData = xBack.dataSync()
  for (let i = 0; i < xData.length; i++) {
    expect(xBackData[i]).toBeCloseTo(xData[i], 3)
  }
  x.dispose()
  y.dispose()
  xBack.dispose()
}

// Helper: check FLDJ + ILDJ = 0 (inverse function theorem)
function checkJacobianConsistency(bijector, points, tol = 1e-3) {
  const x = tf.tensor(points)
  const y = bijector.forward(x)
  const fldj = bijector.forwardLogDetJacobian(x)
  const ildj = bijector.inverseLogDetJacobian(y)
  const fldjData = fldj.dataSync()
  const ildjData = ildj.dataSync()
  for (let i = 0; i < fldjData.length; i++) {
    expect(fldjData[i] + ildjData[i]).toBeCloseTo(0, 2)
  }
  x.dispose()
  y.dispose()
  fldj.dispose()
  ildj.dispose()
}

// Helper: verify FLDJ against finite-difference approximation
function checkFLDJFiniteDifference(bijector, points, eps = 1e-4, tol = 1e-2) {
  for (const x0 of points) {
    const fldj = bijector.forwardLogDetJacobian(x0)
    const fldjVal = fldj.dataSync()[0]
    // Finite difference: dy/dx ≈ (f(x+eps) - f(x-eps)) / (2*eps)
    const yPlus = bijector.forward(x0 + eps)
    const yMinus = bijector.forward(x0 - eps)
    const dydx = (yPlus.dataSync()[0] - yMinus.dataSync()[0]) / (2 * eps)
    const expected = Math.log(Math.abs(dydx))
    expect(fldjVal).toBeCloseTo(expected, 1)
    fldj.dispose()
    yPlus.dispose()
    yMinus.dispose()
  }
}

describe('Identity bijector', () => {
  const b = new Identity()

  test('properties', () => {
    expect(b.name).toBe('Identity')
    expect(b.isConstantJacobian).toBe(true)
  })

  test('forward is identity', () => {
    const y = b.forward(tf.tensor([1, 2, 3]))
    expect(Array.from(y.dataSync())).toEqual([1, 2, 3])
    y.dispose()
  })

  test('inverse is identity', () => {
    const x = b.inverse(tf.tensor([1, 2, 3]))
    expect(Array.from(x.dataSync())).toEqual([1, 2, 3])
    x.dispose()
  })

  test('FLDJ is zero', () => {
    const ldj = b.forwardLogDetJacobian(tf.tensor([1, 2, 3]))
    expect(Array.from(ldj.dataSync())).toEqual([0, 0, 0])
    ldj.dispose()
  })

  test('ILDJ is zero', () => {
    const ldj = b.inverseLogDetJacobian(tf.tensor([1, 2, 3]))
    expect(Array.from(ldj.dataSync())).toEqual([0, 0, 0])
    ldj.dispose()
  })

  test('roundtrip', () => {
    checkRoundtrip(b, [-2, -1, 0, 1, 2])
  })
})

describe('Exp bijector', () => {
  const b = new Exp()

  test('properties', () => {
    expect(b.name).toBe('Exp')
    expect(b.isConstantJacobian).toBe(false)
  })

  test('forward is exp', () => {
    const y = b.forward(tf.tensor([0, 1, 2]))
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(1, 5)
    expect(data[1]).toBeCloseTo(Math.E, 4)
    expect(data[2]).toBeCloseTo(Math.E * Math.E, 3)
    y.dispose()
  })

  test('inverse is log', () => {
    const x = b.inverse(tf.tensor([1, Math.E, Math.E * Math.E]))
    const data = x.dataSync()
    expect(data[0]).toBeCloseTo(0, 4)
    expect(data[1]).toBeCloseTo(1, 4)
    expect(data[2]).toBeCloseTo(2, 3)
    x.dispose()
  })

  test('FLDJ = x', () => {
    const points = [-2, -1, 0, 1, 2]
    const ldj = b.forwardLogDetJacobian(tf.tensor(points))
    const data = ldj.dataSync()
    for (let i = 0; i < points.length; i++) {
      expect(data[i]).toBeCloseTo(points[i], 4)
    }
    ldj.dispose()
  })

  test('ILDJ = -log(y)', () => {
    const points = [0.5, 1, 2, 5]
    const ldj = b.inverseLogDetJacobian(tf.tensor(points))
    const data = ldj.dataSync()
    for (let i = 0; i < points.length; i++) {
      expect(data[i]).toBeCloseTo(-Math.log(points[i]), 4)
    }
    ldj.dispose()
  })

  test('roundtrip', () => {
    checkRoundtrip(b, [-2, -1, 0, 1, 2, 3])
  })

  test('Jacobian consistency', () => {
    checkJacobianConsistency(b, [-2, -1, 0, 1, 2])
  })

  test('FLDJ matches finite difference', () => {
    checkFLDJFiniteDifference(b, [-1, 0, 1, 2])
  })
})

describe('Log bijector', () => {
  const b = new Log()

  test('properties', () => {
    expect(b.name).toBe('Log')
  })

  test('forward is log', () => {
    const y = b.forward(tf.tensor([1, Math.E, 10]))
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(0, 5)
    expect(data[1]).toBeCloseTo(1, 4)
    expect(data[2]).toBeCloseTo(Math.log(10), 4)
    y.dispose()
  })

  test('inverse is exp', () => {
    const x = b.inverse(tf.tensor([0, 1, 2]))
    const data = x.dataSync()
    expect(data[0]).toBeCloseTo(1, 5)
    expect(data[1]).toBeCloseTo(Math.E, 4)
    expect(data[2]).toBeCloseTo(Math.exp(2), 3)
    x.dispose()
  })

  test('FLDJ = -log(x)', () => {
    const points = [0.5, 1, 2, 5]
    const ldj = b.forwardLogDetJacobian(tf.tensor(points))
    const data = ldj.dataSync()
    for (let i = 0; i < points.length; i++) {
      expect(data[i]).toBeCloseTo(-Math.log(points[i]), 4)
    }
    ldj.dispose()
  })

  test('roundtrip', () => {
    checkRoundtrip(b, [0.1, 0.5, 1, 2, 5, 10])
  })

  test('Jacobian consistency', () => {
    checkJacobianConsistency(b, [0.1, 0.5, 1, 2, 5])
  })

  test('Exp and Log are inverses', () => {
    const exp = new Exp()
    const log = new Log()
    const x = tf.tensor([-2, -1, 0, 1, 2])
    const y = exp.forward(x)
    const xBack = log.forward(y)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 4)
    }
    x.dispose()
    y.dispose()
    xBack.dispose()
  })
})

describe('Softplus bijector', () => {
  const b = new Softplus()

  test('properties', () => {
    expect(b.name).toBe('Softplus')
  })

  test('forward is softplus', () => {
    const y = b.forward(tf.tensor([0, 1, 5, -5]))
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(Math.log(2), 4) // softplus(0) = log(2)
    expect(data[1]).toBeCloseTo(Math.log(1 + Math.E), 3) // softplus(1) = log(1+e)
    expect(data[2]).toBeCloseTo(5, 1) // softplus(5) ≈ 5 for large x
    expect(data[3]).toBeCloseTo(Math.log(1 + Math.exp(-5)), 4) // small for negative x
    y.dispose()
  })

  test('inverse roundtrip', () => {
    checkRoundtrip(b, [-3, -1, 0, 1, 3, 5, 10])
  })

  test('FLDJ matches log(sigmoid(x))', () => {
    const points = [-3, -1, 0, 1, 3]
    const x = tf.tensor(points)
    const ldj = b.forwardLogDetJacobian(x)
    const data = ldj.dataSync()
    for (let i = 0; i < points.length; i++) {
      const sigmoid = 1 / (1 + Math.exp(-points[i]))
      expect(data[i]).toBeCloseTo(Math.log(sigmoid), 3)
    }
    ldj.dispose()
    x.dispose()
  })

  test('Jacobian consistency', () => {
    checkJacobianConsistency(b, [-3, -1, 0, 1, 3])
  })

  test('FLDJ matches finite difference', () => {
    checkFLDJFiniteDifference(b, [-2, -1, 0, 1, 2])
  })
})

describe('Sigmoid bijector', () => {
  const b = new Sigmoid()

  test('properties', () => {
    expect(b.name).toBe('Sigmoid')
  })

  test('forward is sigmoid', () => {
    const y = b.forward(tf.tensor([0, -100, 100]))
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(0.5, 5)
    expect(data[1]).toBeCloseTo(0, 3)
    expect(data[2]).toBeCloseTo(1, 3)
    y.dispose()
  })

  test('inverse is logit', () => {
    const x = b.inverse(tf.tensor([0.5, 0.1, 0.9]))
    const data = x.dataSync()
    expect(data[0]).toBeCloseTo(0, 4) // logit(0.5) = 0
    expect(data[1]).toBeCloseTo(Math.log(0.1 / 0.9), 3)
    expect(data[2]).toBeCloseTo(Math.log(0.9 / 0.1), 3)
    x.dispose()
  })

  test('roundtrip', () => {
    checkRoundtrip(b, [-5, -2, -1, 0, 1, 2, 5])
  })

  test('FLDJ is correct', () => {
    const points = [-3, -1, 0, 1, 3]
    const x = tf.tensor(points)
    const ldj = b.forwardLogDetJacobian(x)
    const data = ldj.dataSync()
    for (let i = 0; i < points.length; i++) {
      const s = 1 / (1 + Math.exp(-points[i]))
      const expected = Math.log(s * (1 - s))
      expect(data[i]).toBeCloseTo(expected, 3)
    }
    ldj.dispose()
    x.dispose()
  })

  test('ILDJ is correct', () => {
    const points = [0.1, 0.3, 0.5, 0.7, 0.9]
    const y = tf.tensor(points)
    const ldj = b.inverseLogDetJacobian(y)
    const data = ldj.dataSync()
    for (let i = 0; i < points.length; i++) {
      const expected = -Math.log(points[i]) - Math.log(1 - points[i])
      expect(data[i]).toBeCloseTo(expected, 3)
    }
    ldj.dispose()
    y.dispose()
  })

  test('Jacobian consistency', () => {
    checkJacobianConsistency(b, [-3, -1, 0, 1, 3])
  })

  test('FLDJ matches finite difference', () => {
    checkFLDJFiniteDifference(b, [-2, -1, 0, 1, 2])
  })
})
