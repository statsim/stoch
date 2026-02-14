import * as tf from '@tensorflow/tfjs'
import { leapfrogIntegrate } from '../../../src/mcmc/leapfrog'
import { computeGrads, disposeState } from '../../../src/mcmc/state_util'
import { expectClose } from '../../helpers/tolerance'

/**
 * Helper: create targetLogProbGradFn from a logProb function.
 */
function makeGradFn(logProbFn) {
  return (state) => computeGrads(logProbFn, state)
}

describe('leapfrog integrator', () => {
  describe('1D harmonic oscillator', () => {
    // Target: N(0, 1), so log π(q) = -0.5*q²
    // Hamiltonian: H(q,p) = 0.5*q² + 0.5*p²
    const logProb = (q) => tf.mul(-0.5, tf.square(q))
    const gradFn = makeGradFn(logProb)

    test('approximately conserves energy', () => {
      const q0 = tf.scalar(1.0)
      const p0 = tf.scalar(0.5)

      const H0 = 0.5 * 1.0 * 1.0 + 0.5 * 0.5 * 0.5 // 0.5 + 0.125 = 0.625

      const { finalState, finalMomentum, finalTargetLogProb, finalGrads } =
        leapfrogIntegrate({
          currentState: q0,
          momentum: p0,
          stepSize: 0.1,
          numSteps: 10,
          targetLogProbGradFn: gradFn
        })

      const qf = finalState.dataSync()[0]
      const pf = finalMomentum.dataSync()[0]
      const Hf = 0.5 * qf * qf + 0.5 * pf * pf

      // Energy should be approximately conserved (small ε → small error)
      expectClose(Hf, H0, { atol: 0.01 })

      finalState.dispose()
      finalMomentum.dispose()
      finalTargetLogProb.dispose()
      disposeState(finalGrads)
      q0.dispose()
      p0.dispose()
    })

    test('time reversibility', () => {
      const q0 = tf.scalar(2.0)
      const p0 = tf.scalar(-1.0)

      // Forward integration
      const fwd = leapfrogIntegrate({
        currentState: q0,
        momentum: p0,
        stepSize: 0.1,
        numSteps: 20,
        targetLogProbGradFn: gradFn
      })

      // Negate momentum and integrate back
      const negMom = tf.neg(fwd.finalMomentum)
      const bwd = leapfrogIntegrate({
        currentState: fwd.finalState,
        momentum: negMom,
        stepSize: 0.1,
        numSteps: 20,
        targetLogProbGradFn: gradFn
      })

      // Should return to approximately the original state
      expectClose(bwd.finalState.dataSync()[0], q0.dataSync()[0], { atol: 1e-3 })

      // Clean up
      fwd.finalState.dispose()
      fwd.finalMomentum.dispose()
      fwd.finalTargetLogProb.dispose()
      disposeState(fwd.finalGrads)
      negMom.dispose()
      bwd.finalState.dispose()
      bwd.finalMomentum.dispose()
      bwd.finalTargetLogProb.dispose()
      disposeState(bwd.finalGrads)
      q0.dispose()
      p0.dispose()
    })

    test('returns correct final target log prob', () => {
      const q0 = tf.scalar(1.0)
      const p0 = tf.scalar(0.0)

      const { finalState, finalMomentum, finalTargetLogProb, finalGrads } =
        leapfrogIntegrate({
          currentState: q0,
          momentum: p0,
          stepSize: 0.1,
          numSteps: 5,
          targetLogProbGradFn: gradFn
        })

      const qf = finalState.dataSync()[0]
      const expectedLogProb = -0.5 * qf * qf
      expectClose(finalTargetLogProb.dataSync()[0], expectedLogProb, { atol: 1e-4 })

      finalState.dispose()
      finalMomentum.dispose()
      finalTargetLogProb.dispose()
      disposeState(finalGrads)
      q0.dispose()
      p0.dispose()
    })
  })

  describe('multi-parameter state', () => {
    // Target: independent N(0,1) for both a and b
    const logProb = ({ a, b }) => tf.add(
      tf.mul(-0.5, tf.square(a)),
      tf.mul(-0.5, tf.square(b))
    )
    const gradFn = makeGradFn(logProb)

    test('integrates with object state', () => {
      const state = { a: tf.scalar(1.0), b: tf.scalar(-0.5) }
      const momentum = { a: tf.scalar(0.3), b: tf.scalar(0.7) }

      const H0 = 0.5 * (1.0 + 0.25) + 0.5 * (0.09 + 0.49)

      const { finalState, finalMomentum, finalTargetLogProb, finalGrads } =
        leapfrogIntegrate({
          currentState: state,
          momentum,
          stepSize: 0.1,
          numSteps: 10,
          targetLogProbGradFn: gradFn
        })

      const af = finalState.a.dataSync()[0]
      const bf = finalState.b.dataSync()[0]
      const pa = finalMomentum.a.dataSync()[0]
      const pb = finalMomentum.b.dataSync()[0]
      const Hf = 0.5 * (af * af + bf * bf) + 0.5 * (pa * pa + pb * pb)

      expectClose(Hf, H0, { atol: 0.01 })

      disposeState(finalState)
      disposeState(finalMomentum)
      finalTargetLogProb.dispose()
      disposeState(finalGrads)
      disposeState(state)
      disposeState(momentum)
    })
  })

  describe('with pre-computed grads', () => {
    const logProb = (q) => tf.mul(-0.5, tf.square(q))
    const gradFn = makeGradFn(logProb)

    test('uses provided initial grads', () => {
      const q0 = tf.scalar(1.0)
      const p0 = tf.scalar(0.5)
      const initialGrads = tf.scalar(-1.0) // ∇(-0.5*q²) at q=1 is -1
      const initialLogProb = tf.scalar(-0.5) // -0.5*1² = -0.5

      const { finalState, finalMomentum, finalTargetLogProb, finalGrads } =
        leapfrogIntegrate({
          currentState: q0,
          momentum: p0,
          stepSize: 0.1,
          numSteps: 5,
          targetLogProbGradFn: gradFn,
          currentTargetLogProb: initialLogProb,
          currentGrads: initialGrads
        })

      // Should produce valid results
      expect(isFinite(finalState.dataSync()[0])).toBe(true)
      expect(isFinite(finalMomentum.dataSync()[0])).toBe(true)

      finalState.dispose()
      finalMomentum.dispose()
      finalTargetLogProb.dispose()
      disposeState(finalGrads)
      q0.dispose()
      p0.dispose()
      initialGrads.dispose()
      // initialLogProb is reused by leapfrog when numSteps=0, but we passed it
      // and leapfrog replaces it with the final step's value
    })
  })
})
