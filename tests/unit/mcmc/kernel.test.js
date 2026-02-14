import { TransitionKernel } from '../../../src/mcmc/kernel'

describe('TransitionKernel', () => {
  test('oneStep throws by default', () => {
    const kernel = new TransitionKernel()
    expect(() => kernel.oneStep()).toThrow('not implemented')
  })

  test('bootstrapResults throws by default', () => {
    const kernel = new TransitionKernel()
    expect(() => kernel.bootstrapResults()).toThrow('not implemented')
  })

  test('isCalibrated defaults to true', () => {
    const kernel = new TransitionKernel()
    expect(kernel.isCalibrated).toBe(true)
  })
})
