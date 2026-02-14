import * as tf from '@tensorflow/tfjs'
import { Power } from '../../../src/bijectors/power'

describe('Power bijector', () => {
  test('forward: x^2', () => {
    const b = new Power({ power: 2 })
    const y = b.forward(tf.tensor([1, 2, 3]))
    const data = y.dataSync()
    expect(data[0]).toBeCloseTo(1, 5)
    expect(data[1]).toBeCloseTo(4, 5)
    expect(data[2]).toBeCloseTo(9, 5)
    y.dispose()
    b.dispose()
  })

  test('inverse: y^(1/2)', () => {
    const b = new Power({ power: 2 })
    const x = b.inverse(tf.tensor([1, 4, 9]))
    const data = x.dataSync()
    expect(data[0]).toBeCloseTo(1, 5)
    expect(data[1]).toBeCloseTo(2, 5)
    expect(data[2]).toBeCloseTo(3, 5)
    x.dispose()
    b.dispose()
  })

  test('power=1 is identity', () => {
    const b = new Power({ power: 1 })
    const x = tf.tensor([1, 2, 3])
    const y = b.forward(x)
    const data = y.dataSync()
    for (let i = 0; i < 3; i++) {
      expect(data[i]).toBeCloseTo(x.dataSync()[i], 5)
    }
    x.dispose(); y.dispose()
    b.dispose()
  })

  test('forward-inverse roundtrip', () => {
    const b = new Power({ power: 3 })
    const x = tf.tensor([0.5, 1, 2, 3])
    const y = b.forward(x)
    const xBack = b.inverse(y)
    const xData = x.dataSync()
    const xBackData = xBack.dataSync()
    for (let i = 0; i < xData.length; i++) {
      expect(xBackData[i]).toBeCloseTo(xData[i], 4)
    }
    x.dispose(); y.dispose(); xBack.dispose()
    b.dispose()
  })

  test('FLDJ + ILDJ = 0', () => {
    const b = new Power({ power: 2 })
    const x = tf.tensor([0.5, 1, 2])
    const y = b.forward(x)
    const fldj = b.forwardLogDetJacobian(x)
    const ildj = b.inverseLogDetJacobian(y)
    const sum = tf.add(fldj, ildj).dataSync()
    for (let i = 0; i < sum.length; i++) {
      expect(sum[i]).toBeCloseTo(0, 4)
    }
    x.dispose(); y.dispose(); fldj.dispose(); ildj.dispose()
    b.dispose()
  })

  test('FLDJ for x^2: log(2) + log(x)', () => {
    const b = new Power({ power: 2 })
    const x = tf.tensor([1, 2, 3])
    const fldj = b.forwardLogDetJacobian(x)
    const data = fldj.dataSync()
    expect(data[0]).toBeCloseTo(Math.log(2) + Math.log(1), 4)
    expect(data[1]).toBeCloseTo(Math.log(2) + Math.log(2), 4)
    expect(data[2]).toBeCloseTo(Math.log(2) + Math.log(3), 4)
    fldj.dispose()
    b.dispose()
  })
})
