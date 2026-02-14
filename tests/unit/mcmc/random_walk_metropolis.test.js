import * as tf from '@tensorflow/tfjs'
import { RandomWalkMetropolis } from '../../../src/mcmc/random_walk_metropolis'
import { sampleChain } from '../../../src/mcmc/sample_chain'
import { disposeState } from '../../../src/mcmc/state_util'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('RandomWalkMetropolis', () => {
  const normalLogProb = (x) => tf.mul(-0.5, tf.square(x))

  test('basic oneStep', () => {
    const kernel = new RandomWalkMetropolis({
      targetLogProbFn: normalLogProb,
      proposalScale: 0.5
    })

    const state = tf.scalar(0)
    const kr = kernel.bootstrapResults(state)

    const { nextState, kernelResults } = kernel.oneStep(state, kr)
    expect(nextState instanceof tf.Tensor).toBe(true)
    expect(isFinite(nextState.dataSync()[0])).toBe(true)

    nextState.dispose()
    kernelResults.targetLogProb.dispose()
    kernelResults.isAccepted.dispose()
    kernelResults.logAcceptRatio.dispose()
    kr.targetLogProb.dispose()
    kr.isAccepted.dispose()
    kr.logAcceptRatio.dispose()
    state.dispose()
  })

  test('recovers N(0,1) mean with sampleChain', () => {
    const kernel = new RandomWalkMetropolis({
      targetLogProbFn: normalLogProb,
      proposalScale: 1.0
    })

    const { samples } = sampleChain({
      numResults: 1000,
      numBurninSteps: 500,
      currentState: tf.scalar(0),
      kernel
    })

    const stats = sampleStats(samples.dataSync())
    expectClose(stats.mean, 0, { atol: 0.3 })

    samples.dispose()
  })

  test('multi-parameter state', () => {
    const kernel = new RandomWalkMetropolis({
      targetLogProbFn: ({ a, b }) => tf.tidy(() =>
        tf.add(tf.mul(-0.5, tf.square(a)), tf.mul(-0.5, tf.square(b)))
      ),
      proposalScale: 0.5
    })

    const { samples } = sampleChain({
      numResults: 1000,
      numBurninSteps: 500,
      currentState: { a: tf.scalar(0), b: tf.scalar(0) },
      kernel
    })

    const aStats = sampleStats(samples.a.dataSync())
    expectClose(aStats.mean, 0, { atol: 0.4 })

    samples.a.dispose()
    samples.b.dispose()
  })

  test('custom proposal function', () => {
    const kernel = new RandomWalkMetropolis({
      targetLogProbFn: normalLogProb,
      newStateProposalFn: (state) => tf.tidy(() =>
        tf.add(state, tf.randomNormal(state.shape, 0, 2.0))
      )
    })

    const { samples } = sampleChain({
      numResults: 300,
      numBurninSteps: 100,
      currentState: tf.scalar(0),
      kernel
    })

    expect(samples.shape).toEqual([300])
    samples.dispose()
  })
})
