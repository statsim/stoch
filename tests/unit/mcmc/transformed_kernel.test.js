import * as tf from '@tensorflow/tfjs'
import { TransformedTransitionKernel } from '../../../src/mcmc/transformed_kernel'
import { HamiltonianMonteCarlo } from '../../../src/mcmc/hmc'
import { sampleChain } from '../../../src/mcmc/sample_chain'
import { Exp } from '../../../src/bijectors/exp'
import { disposeState } from '../../../src/mcmc/state_util'
import { expectClose, sampleStats } from '../../helpers/tolerance'

describe('TransformedTransitionKernel', () => {
  describe('single-tensor state with Exp bijector', () => {
    test('samples positive values from LogNormal-like target', () => {
      // Target in constrained space: log(sigma) ~ N(0,1)
      // i.e., sigma ~ LogNormal(0,1), so sigma > 0
      // In constrained space: logπ(sigma) = -0.5*(log(sigma))² - log(sigma)
      const constrainedLogProb = (sigma) => tf.tidy(() => {
        const logSigma = tf.log(sigma)
        return tf.sub(tf.mul(-0.5, tf.square(logSigma)), logSigma)
      })

      const kernel = new TransformedTransitionKernel({
        innerKernel: new HamiltonianMonteCarlo({
          targetLogProbFn: constrainedLogProb, // will be wrapped
          stepSize: 0.3,
          numLeapfrogSteps: 5
        }),
        bijectors: new Exp()
      })

      const { samples } = sampleChain({
        numResults: 200,
        numBurninSteps: 100,
        currentState: tf.scalar(1.0), // constrained (positive)
        kernel
      })

      const data = samples.dataSync()
      // All samples should be positive (constrained space)
      for (let i = 0; i < data.length; i++) {
        expect(data[i]).toBeGreaterThan(0)
      }

      samples.dispose()
    })
  })

  describe('object state with partial bijectors', () => {
    test('constrains only specified parameters', () => {
      // Model: mu ~ N(0,1), sigma ~ LogNormal(0,1)
      // Only sigma needs Exp bijector
      const constrainedLogProb = ({ mu, sigma }) => tf.tidy(() => {
        const muPart = tf.mul(-0.5, tf.square(mu))
        const logSigma = tf.log(sigma)
        const sigmaPart = tf.sub(tf.mul(-0.5, tf.square(logSigma)), logSigma)
        return tf.add(muPart, sigmaPart)
      })

      const kernel = new TransformedTransitionKernel({
        innerKernel: new HamiltonianMonteCarlo({
          targetLogProbFn: constrainedLogProb,
          stepSize: 0.2,
          numLeapfrogSteps: 5
        }),
        bijectors: { sigma: new Exp() }
      })

      const { samples } = sampleChain({
        numResults: 200,
        numBurninSteps: 100,
        currentState: { mu: tf.scalar(0), sigma: tf.scalar(1) },
        kernel
      })

      const sigmaData = samples.sigma.dataSync()
      // All sigma samples should be positive
      for (let i = 0; i < sigmaData.length; i++) {
        expect(sigmaData[i]).toBeGreaterThan(0)
      }

      // mu should be roughly centered at 0
      const muStats = sampleStats(samples.mu.dataSync())
      expectClose(muStats.mean, 0, { atol: 0.5 })

      samples.mu.dispose()
      samples.sigma.dispose()
    })
  })
})
