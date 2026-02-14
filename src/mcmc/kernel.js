/**
 * Base class for MCMC transition kernels.
 *
 * A TransitionKernel takes a current state and produces a new state,
 * along with diagnostic information (kernel results).
 *
 * Subclasses must implement:
 *   oneStep(currentState, previousKernelResults)
 *     → { nextState, kernelResults }
 *   bootstrapResults(initState)
 *     → initial kernelResults
 *
 * Convention:
 *   - currentState is a Tensor or object-of-Tensors
 *   - kernelResults is a plain object with diagnostics
 *   - isCalibrated indicates whether the kernel targets the exact distribution
 */
export class TransitionKernel {
  /**
   * Take one MCMC step.
   * @param {Object|tf.Tensor} currentState
   * @param {Object} previousKernelResults
   * @returns {{ nextState: Object|tf.Tensor, kernelResults: Object }}
   */
  oneStep(currentState, previousKernelResults) {
    throw new Error('TransitionKernel.oneStep not implemented')
  }

  /**
   * Create initial kernel results from an initial state.
   * @param {Object|tf.Tensor} initState
   * @returns {Object}
   */
  bootstrapResults(initState) {
    throw new Error('TransitionKernel.bootstrapResults not implemented')
  }

  /**
   * Whether this kernel produces exact samples from the target distribution.
   * Calibrated kernels (e.g., HMC with MH correction) satisfy detailed balance.
   */
  get isCalibrated() {
    return true
  }
}
