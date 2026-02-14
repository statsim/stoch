import * as tf from '@tensorflow/tfjs'
import { DualAveragingStepSizeAdaptation } from '../../../src/mcmc/dual_averaging'
import { HamiltonianMonteCarlo } from '../../../src/mcmc/hmc'
import { sampleChain } from '../../../src/mcmc/sample_chain'
import { disposeState } from '../../../src/mcmc/state_util'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('DualAveragingStepSizeAdaptation', () => {
  const normalLogProb = (x) => tf.mul(-0.5, tf.square(x))

  test('adapts step size during warmup', () => {
    const innerKernel = new HamiltonianMonteCarlo({
      targetLogProbFn: normalLogProb,
      stepSize: 5.0, // deliberately bad initial step size
      numLeapfrogSteps: 5
    })

    const kernel = new DualAveragingStepSizeAdaptation({
      innerKernel,
      numAdaptationSteps: 50,
      targetAcceptProb: 0.75
    })

    const initState = tf.scalar(0)
    let state = initState.clone()
    let kr = kernel.bootstrapResults(state)

    // Run adaptation steps
    for (let i = 0; i < 60; i++) {
      const { nextState, kernelResults } = kernel.oneStep(state, kr)

      // Dispose previous
      state.dispose()
      if (kr.innerResults) {
        kr.innerResults.targetLogProb.dispose()
        disposeState(kr.innerResults.gradsTargetLogProb)
        kr.innerResults.isAccepted.dispose()
        kr.innerResults.logAcceptRatio.dispose()
        kr.innerResults.stepSize.dispose()
      }

      state = nextState
      kr = kernelResults
    }

    // Step size should have been adapted (not still 5.0)
    const finalStepSize = innerKernel._stepSize
    expect(finalStepSize).not.toBe(5.0)
    expect(finalStepSize).toBeGreaterThan(0)
    expect(finalStepSize).toBeLessThan(5.0) // should decrease from bad initial

    // Clean up
    state.dispose()
    kr.innerResults.targetLogProb.dispose()
    disposeState(kr.innerResults.gradsTargetLogProb)
    kr.innerResults.isAccepted.dispose()
    kr.innerResults.logAcceptRatio.dispose()
    kr.innerResults.stepSize.dispose()
    initState.dispose()
  })

  test('works with sampleChain', () => {
    const innerKernel = new HamiltonianMonteCarlo({
      targetLogProbFn: normalLogProb,
      stepSize: 1.0,
      numLeapfrogSteps: 10
    })

    const kernel = new DualAveragingStepSizeAdaptation({
      innerKernel,
      numAdaptationSteps: 100,
      targetAcceptProb: 0.75
    })

    const { samples } = sampleChain({
      numResults: 300,
      numBurninSteps: 200, // includes adaptation
      currentState: tf.scalar(0),
      kernel
    })

    const stats = sampleStats(samples.dataSync())
    expectClose(stats.mean, 0, { atol: 0.4 })
    expectClose(stats.variance, 1, { atol: 0.6 })

    samples.dispose()
  })
})
