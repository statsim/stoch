import * as tf from '@tensorflow/tfjs'
import { TransitionKernel } from './kernel'
import { disposeState } from './state_util'

/**
 * Dual averaging step size adaptation for HMC/NUTS.
 *
 * Implements Nesterov's dual averaging algorithm
 * (Hoffman & Gelman 2014, Algorithm 5; same as Stan).
 * Adjusts the step size during a warmup period to achieve a target
 * acceptance probability, then freezes.
 *
 * @param {Object} params
 * @param {TransitionKernel} params.innerKernel - HMC/NUTS kernel (or wrapped)
 * @param {number} [params.numAdaptationSteps=400]
 * @param {number} [params.targetAcceptProb=0.75]
 * @param {number} [params.shrinkageTarget] - μ, defaults to log(10 * initial step size)
 * @param {number} [params.stepCountSmoothing=10] - t₀
 * @param {number} [params.decayRate=0.75] - κ (for smoothed average)
 * @param {number} [params.gamma=0.05] - γ (adaptation regularization)
 */
export class DualAveragingStepSizeAdaptation extends TransitionKernel {
  constructor({
    innerKernel,
    numAdaptationSteps = 400,
    targetAcceptProb = 0.75,
    shrinkageTarget,
    stepCountSmoothing = 10,
    decayRate = 0.75,
    gamma = 0.05
  }) {
    super()
    this._innerKernel = innerKernel
    this._numAdaptationSteps = numAdaptationSteps
    this._targetAcceptProb = targetAcceptProb
    this._stepCountSmoothing = stepCountSmoothing
    this._decayRate = decayRate
    this._gamma = gamma

    // Find the kernel that owns _stepSize, traversing through wrappers
    // (e.g. TransformedTransitionKernel) that don't have their own step size.
    this._stepSizeKernel = this._findStepSizeKernel(innerKernel)

    // Get initial step size
    const initEps = this._getStepSize()
    this._shrinkageTarget = shrinkageTarget || Math.log(10 * initEps)
    this._logEpsBar = Math.log(initEps)
    this._hBar = 0
  }

  get innerKernel() { return this._innerKernel }

  /**
   * Traverse through wrapper kernels to find the one with _stepSize.
   * Supports TransformedTransitionKernel and other wrappers that
   * expose their inner kernel via _innerKernel.
   */
  _findStepSizeKernel(kernel) {
    if (kernel._stepSize !== undefined) return kernel
    if (kernel._innerKernel) return this._findStepSizeKernel(kernel._innerKernel)
    throw new Error('DualAveragingStepSizeAdaptation: could not find _stepSize on inner kernel chain')
  }

  _getStepSize() {
    const ss = this._stepSizeKernel._stepSize
    return typeof ss === 'number' ? ss : ss.dataSync()[0]
  }

  _setStepSize(value) {
    this._stepSizeKernel._stepSize = value
  }

  bootstrapResults(initState) {
    const innerKR = this._innerKernel.bootstrapResults(initState)
    return {
      innerResults: innerKR,
      step: 0,
      logEps: Math.log(this._getStepSize()),
      logEpsBar: this._logEpsBar,
      hBar: this._hBar
    }
  }

  oneStep(currentState, previousKernelResults) {
    const step = previousKernelResults.step
    const isAdapting = step < this._numAdaptationSteps

    // Run the inner kernel
    const { nextState, kernelResults: innerKR } =
      this._innerKernel.oneStep(currentState, previousKernelResults.innerResults)

    let logEps = previousKernelResults.logEps
    let logEpsBar = previousKernelResults.logEpsBar
    let hBar = previousKernelResults.hBar

    if (isAdapting) {
      // Extract acceptance probability — dig through wrapper layers
      let ratioKR = innerKR
      while (ratioKR.innerResults) ratioKR = ratioKR.innerResults
      const logAcceptRatio = ratioKR.logAcceptRatio.dataSync()[0]
      // Treat NaN as rejection (divergent proposal)
      const acceptProb = isNaN(logAcceptRatio) ? 0 : Math.min(1, Math.exp(logAcceptRatio))

      // Dual averaging update (Hoffman & Gelman 2014, Algorithm 5)
      const m = step + 1
      const t0 = this._stepCountSmoothing
      const kappa = this._decayRate
      const gamma = this._gamma
      const mu = this._shrinkageTarget
      const delta = this._targetAcceptProb

      // Update H̄: running average of (δ - α_m)
      hBar = (1 - 1 / (m + t0)) * hBar + (delta - acceptProb) / (m + t0)

      // Dual variable: log ε_m = μ − √m / γ · H̄_m
      logEps = mu - Math.sqrt(m) / gamma * hBar

      // Smoothed average: log ε̄_m = m^{-κ} log ε_m + (1 − m^{-κ}) log ε̄_{m−1}
      const eta = m ** (-kappa)
      logEpsBar = eta * logEps + (1 - eta) * logEpsBar

      // Update inner kernel step size
      this._setStepSize(Math.exp(logEps))
    } else if (step === this._numAdaptationSteps) {
      // Freeze: use the smoothed average
      this._setStepSize(Math.exp(logEpsBar))
      logEps = logEpsBar
    }

    return {
      nextState,
      kernelResults: {
        innerResults: innerKR,
        step: step + 1,
        logEps,
        logEpsBar,
        hBar
      }
    }
  }
}
