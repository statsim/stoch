import * as tf from '@tensorflow/tfjs'
import { Shift } from '../../../src/bijectors/shift'
import { Scale } from '../../../src/bijectors/scale'
import { Chain } from '../../../src/bijectors/chain'
import { Exp } from '../../../src/bijectors/exp'
import { Softplus } from '../../../src/bijectors/softplus'

describe('Shift bijector', () => {
  test('properties', () => {
    const b = new Shift({ shift: 3 })
    expect(b.name).toBe('Shift')
    expect(b.isConstantJacobian).toBe(true)
    b.dispose()
  })

  test('forward adds shift', () => {
    const b = new Shift({ shift: 5 })
    const y = b.forward(tf.tensor([1, 2, 3]))
    expect(Array.from(y.dataSync())).toEqual([6, 7, 8])
    y.dispose()
    b.dispose()
  })

  test('inverse subtracts shift', () => {
    const b = new Shift({ shift: 5 })
    const x = b.inverse(tf.tensor([6, 7, 8]))
    expect(Array.from(x.dataSync())).toEqual([1, 2, 3])
    x.dispose()
    b.dispose()
  })

  test('FLDJ is zero', () => {
    const b = new Shift({ shift: 5 })
    const ldj = b.forwardLogDetJacobian(tf.tensor([1, 2, 3]))
    expect(Array.from(ldj.dataSync())).toEqual([0, 0, 0])
    ldj.dispose()
    b.dispose()
  })

  test('roundtrip', () => {
    const b = new Shift({ shift: -7 })
    const x = tf.tensor([-3, 0, 5])
    const y = b.forward(x)
    const xBack = b.inverse(y)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 5)
    }
    x.dispose()
    y.dispose()
    xBack.dispose()
    b.dispose()
  })

  test('dispose frees shift tensor', () => {
    const before = tf.memory().numTensors
    const b = new Shift({ shift: 3 })
    expect(tf.memory().numTensors).toBeGreaterThan(before)
    b.dispose()
    expect(tf.memory().numTensors).toBe(before)
  })
})

describe('Scale bijector', () => {
  test('properties', () => {
    const b = new Scale({ scale: 2 })
    expect(b.name).toBe('Scale')
    expect(b.isConstantJacobian).toBe(true)
    b.dispose()
  })

  test('forward multiplies by scale', () => {
    const b = new Scale({ scale: 3 })
    const y = b.forward(tf.tensor([1, 2, 3]))
    expect(Array.from(y.dataSync())).toEqual([3, 6, 9])
    y.dispose()
    b.dispose()
  })

  test('inverse divides by scale', () => {
    const b = new Scale({ scale: 2 })
    const x = b.inverse(tf.tensor([4, 6, 8]))
    expect(Array.from(x.dataSync())).toEqual([2, 3, 4])
    x.dispose()
    b.dispose()
  })

  test('FLDJ = log|scale|', () => {
    const b = new Scale({ scale: 3 })
    const ldj = b.forwardLogDetJacobian(tf.tensor([1, 2]))
    const data = ldj.dataSync()
    expect(data[0]).toBeCloseTo(Math.log(3), 4)
    expect(data[1]).toBeCloseTo(Math.log(3), 4)
    ldj.dispose()
    b.dispose()
  })

  test('ILDJ = -log|scale|', () => {
    const b = new Scale({ scale: 3 })
    const ldj = b.inverseLogDetJacobian(tf.tensor([1, 2]))
    const data = ldj.dataSync()
    expect(data[0]).toBeCloseTo(-Math.log(3), 4)
    expect(data[1]).toBeCloseTo(-Math.log(3), 4)
    ldj.dispose()
    b.dispose()
  })

  test('negative scale works', () => {
    const b = new Scale({ scale: -2 })
    const y = b.forward(tf.tensor([1, 2, 3]))
    expect(Array.from(y.dataSync())).toEqual([-2, -4, -6])
    const ldj = b.forwardLogDetJacobian(tf.tensor([1]))
    expect(ldj.dataSync()[0]).toBeCloseTo(Math.log(2), 4) // log|scale|
    y.dispose()
    ldj.dispose()
    b.dispose()
  })

  test('roundtrip', () => {
    const b = new Scale({ scale: 0.5 })
    const x = tf.tensor([-3, 0, 5])
    const y = b.forward(x)
    const xBack = b.inverse(y)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 5)
    }
    x.dispose()
    y.dispose()
    xBack.dispose()
    b.dispose()
  })

  test('Jacobian consistency (FLDJ + ILDJ = 0)', () => {
    const b = new Scale({ scale: 5 })
    const x = tf.tensor([1, 2, 3])
    const y = b.forward(x)
    const fldj = b.forwardLogDetJacobian(x)
    const ildj = b.inverseLogDetJacobian(y)
    const fldjData = fldj.dataSync()
    const ildjData = ildj.dataSync()
    for (let i = 0; i < fldjData.length; i++) {
      expect(fldjData[i] + ildjData[i]).toBeCloseTo(0, 4)
    }
    x.dispose()
    y.dispose()
    fldj.dispose()
    ildj.dispose()
    b.dispose()
  })
})

describe('Chain bijector', () => {
  test('properties', () => {
    const c = new Chain({ bijectors: [new Exp(), new Scale({ scale: 2 })] })
    expect(c.name).toBe('Chain')
    expect(c.bijectors.length).toBe(2)
    c.dispose()
  })

  test('constant Jacobian when all components are constant', () => {
    const c = new Chain({ bijectors: [new Shift({ shift: 1 }), new Scale({ scale: 2 })] })
    expect(c.isConstantJacobian).toBe(true)
    c.dispose()
  })

  test('non-constant Jacobian when any component is non-constant', () => {
    const c = new Chain({ bijectors: [new Exp(), new Scale({ scale: 2 })] })
    expect(c.isConstantJacobian).toBe(false)
    c.dispose()
  })

  test('forward applies right-to-left: Scale then Exp', () => {
    // Chain([Exp, Scale(2)]): forward does Scale first, then Exp
    // forward(x) = Exp(Scale(x)) = exp(2x)
    const c = new Chain({ bijectors: [new Exp(), new Scale({ scale: 2 })] })
    const y = c.forward(tf.tensor([0, 1, -1]))
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(Math.exp(0), 4)  // exp(2*0) = 1
    expect(data[1]).toBeCloseTo(Math.exp(2), 3)  // exp(2*1)
    expect(data[2]).toBeCloseTo(Math.exp(-2), 3) // exp(2*(-1))
    y.dispose()
    c.dispose()
  })

  test('inverse applies left-to-right: Exp⁻¹ then Scale⁻¹', () => {
    const c = new Chain({ bijectors: [new Exp(), new Scale({ scale: 2 })] })
    // inverse(y) = Scale⁻¹(Exp⁻¹(y)) = log(y) / 2
    const x = c.inverse(tf.tensor([1, Math.exp(2), Math.exp(-2)]))
    const data = x.dataSync()
    expect(data[0]).toBeCloseTo(0, 4)  // log(1)/2 = 0
    expect(data[1]).toBeCloseTo(1, 3)  // log(exp(2))/2 = 1
    expect(data[2]).toBeCloseTo(-1, 3) // log(exp(-2))/2 = -1
    x.dispose()
    c.dispose()
  })

  test('roundtrip', () => {
    const c = new Chain({ bijectors: [new Exp(), new Scale({ scale: 2 })] })
    const x = tf.tensor([-2, -1, 0, 1, 2])
    const y = c.forward(x)
    const xBack = c.inverse(y)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 3)
    }
    x.dispose()
    y.dispose()
    xBack.dispose()
    c.dispose()
  })

  test('FLDJ is sum of component FLDJs', () => {
    // Chain([Exp, Scale(2)]): forward(x) = exp(2x)
    // FLDJ = log|d(exp(2x))/dx| = log(2 * exp(2x)) = log(2) + 2x
    const c = new Chain({ bijectors: [new Exp(), new Scale({ scale: 2 })] })
    const points = [-1, 0, 1, 2]
    const x = tf.tensor(points)
    const ldj = c.forwardLogDetJacobian(x)
    const data = ldj.dataSync()
    for (let i = 0; i < points.length; i++) {
      const expected = Math.log(2) + 2 * points[i]
      expect(data[i]).toBeCloseTo(expected, 3)
    }
    ldj.dispose()
    x.dispose()
    c.dispose()
  })

  test('ILDJ is negative of FLDJ at corresponding points', () => {
    const c = new Chain({ bijectors: [new Exp(), new Scale({ scale: 2 })] })
    const x = tf.tensor([-1, 0, 1])
    const y = c.forward(x)
    const fldj = c.forwardLogDetJacobian(x)
    const ildj = c.inverseLogDetJacobian(y)
    const fldjData = fldj.dataSync()
    const ildjData = ildj.dataSync()
    for (let i = 0; i < fldjData.length; i++) {
      expect(fldjData[i] + ildjData[i]).toBeCloseTo(0, 2)
    }
    x.dispose()
    y.dispose()
    fldj.dispose()
    ildj.dispose()
    c.dispose()
  })

  test('chain of three bijectors', () => {
    // Chain([Exp, Shift(1), Scale(2)]): forward(x) = exp(2x + 1)
    const c = new Chain({
      bijectors: [new Exp(), new Shift({ shift: 1 }), new Scale({ scale: 2 })]
    })
    const y = c.forward(tf.scalar(0))
    expect(y.dataSync()[0]).toBeCloseTo(Math.exp(1), 3) // exp(2*0 + 1) = e
    y.dispose()

    const x = c.inverse(tf.scalar(Math.E))
    expect(x.dataSync()[0]).toBeCloseTo(0, 3) // (log(e) - 1) / 2 = 0
    x.dispose()
    c.dispose()
  })

  test('affine chain (shift + scale) roundtrip', () => {
    // y = 2x + 3
    const c = new Chain({ bijectors: [new Shift({ shift: 3 }), new Scale({ scale: 2 })] })
    const x = tf.tensor([0, 1, -1, 5])
    const y = c.forward(x)
    const expected = [3, 5, 1, 13]
    const data = y.dataSync()
    for (let i = 0; i < expected.length; i++) {
      expect(data[i]).toBeCloseTo(expected[i], 5)
    }
    const xBack = c.inverse(y)
    const xBackData = xBack.dataSync()
    const xData = x.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 5)
    }
    x.dispose()
    y.dispose()
    xBack.dispose()
    c.dispose()
  })

  test('empty chain is identity', () => {
    const c = new Chain({ bijectors: [] })
    const y = c.forward(tf.tensor([1, 2, 3]))
    expect(Array.from(y.dataSync())).toEqual([1, 2, 3])
    y.dispose()

    const ldj = c.forwardLogDetJacobian(tf.tensor([1]))
    expect(ldj.dataSync()[0]).toBeCloseTo(0, 5)
    ldj.dispose()
    c.dispose()
  })
})
