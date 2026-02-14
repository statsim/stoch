import { LOG_PI, LOG_2, LOG_2PI, LOG_SQRT_2PI, SQRT_2 } from '../../../src/math/numeric'

describe('math/numeric constants', () => {
  test('LOG_PI', () => {
    expect(LOG_PI).toBeCloseTo(Math.log(Math.PI), 14)
  })

  test('LOG_2', () => {
    expect(LOG_2).toBeCloseTo(Math.log(2), 14)
  })

  test('LOG_2PI', () => {
    expect(LOG_2PI).toBeCloseTo(Math.log(2 * Math.PI), 14)
  })

  test('LOG_SQRT_2PI', () => {
    expect(LOG_SQRT_2PI).toBeCloseTo(0.5 * Math.log(2 * Math.PI), 14)
  })

  test('SQRT_2', () => {
    expect(SQRT_2).toBeCloseTo(Math.SQRT2, 14)
  })
})
