import * as tf from '@tensorflow/tfjs'
import { GaussianProcess } from '../../../src/gp/gaussian_process'
import { GaussianProcessRegressionModel } from '../../../src/gp/gp_regression'
import { SquaredExponential } from '../../../src/gp/kernels/squared_exponential'
import { White } from '../../../src/gp/kernels/white'
import { Add } from '../../../src/gp/kernels/combinators'

describe('GaussianProcess', () => {
  test('sample returns correct shape', () => {
    const gp = new GaussianProcess({
      kernel: new SquaredExponential()
    })
    const x = tf.tensor2d([[0], [1], [2], [3], [4]])
    const s = gp.sample(x)
    expect(s.shape).toEqual([5])
    s.dispose(); x.dispose()
  })

  test('sample with sampleShape', () => {
    const gp = new GaussianProcess({
      kernel: new SquaredExponential()
    })
    const x = tf.tensor2d([[0], [1], [2]])
    const s = gp.sample(x, [10])
    expect(s.shape).toEqual([10, 3])
    s.dispose(); x.dispose()
  })

  test('logProb returns finite scalar', () => {
    const gp = new GaussianProcess({
      kernel: new SquaredExponential(),
      observationNoiseVariance: 0.1
    })
    const x = tf.tensor2d([[0], [1], [2]])
    const y = tf.tensor1d([0.1, 0.9, 2.1])
    const lp = gp.logProb(x, y)
    expect(lp.shape).toEqual([])
    expect(isFinite(lp.dataSync()[0])).toBe(true)
    lp.dispose(); x.dispose(); y.dispose()
  })

  test('posterior mean interpolates observations (no noise)', () => {
    const gp = new GaussianProcess({
      kernel: new SquaredExponential({ lengthScale: 1 }),
      observationNoiseVariance: 1e-6
    })
    const xTrain = tf.tensor2d([[0], [1], [2]])
    const yTrain = tf.tensor1d([0, 1, 0])
    const xTest = tf.tensor2d([[0], [1], [2]])

    const { mean, covariance } = gp.posterior(xTrain, yTrain, xTest)

    // Posterior mean at training points should match observations
    const m = mean.dataSync()
    expect(m[0]).toBeCloseTo(0, 1)
    expect(m[1]).toBeCloseTo(1, 1)
    expect(m[2]).toBeCloseTo(0, 1)

    // Posterior variance at training points should be near zero
    const cov = covariance.dataSync()
    expect(cov[0]).toBeLessThan(0.01)
    expect(cov[4]).toBeLessThan(0.01)

    mean.dispose(); covariance.dispose()
    xTrain.dispose(); yTrain.dispose(); xTest.dispose()
  })

  test('posterior uncertainty increases far from data', () => {
    const gp = new GaussianProcess({
      kernel: new SquaredExponential({ lengthScale: 1 }),
      observationNoiseVariance: 1e-6
    })
    const xTrain = tf.tensor2d([[0]])
    const yTrain = tf.tensor1d([1])
    const xTest = tf.tensor2d([[0], [5], [10]])

    const { mean, covariance } = gp.posterior(xTrain, yTrain, xTest)

    const cov = covariance.dataSync()
    const varNear = cov[0]   // variance at x=0 (near training point)
    const varFar = cov[8]    // variance at x=10 (far from training point)
    expect(varFar).toBeGreaterThan(varNear)

    mean.dispose(); covariance.dispose()
    xTrain.dispose(); yTrain.dispose(); xTest.dispose()
  })

  test('custom mean function', () => {
    const gp = new GaussianProcess({
      kernel: new SquaredExponential(),
      meanFn: (x) => tf.fill([x.shape[0]], 5) // constant mean = 5
    })
    const x = tf.tensor2d([[0], [1]])
    const s = gp.sample(x)
    // Samples should be centered around 5
    expect(s.shape).toEqual([2])
    s.dispose(); x.dispose()
  })
})

describe('GaussianProcessRegressionModel', () => {
  test('predict returns mean and covariance', () => {
    const xTrain = tf.tensor2d([[0], [1], [2]])
    const yTrain = tf.tensor1d([0, 1, 0])
    const xTest = tf.tensor2d([[0.5], [1.5]])

    const gprm = new GaussianProcessRegressionModel({
      kernel: new SquaredExponential(),
      indexPoints: xTrain,
      observations: yTrain,
      observationNoiseVariance: 0.01
    })

    const { mean, covariance } = gprm.predict(xTest)
    expect(mean.shape).toEqual([2])
    expect(covariance.shape).toEqual([2, 2])

    mean.dispose(); covariance.dispose()
    xTrain.dispose(); yTrain.dispose(); xTest.dispose()
  })

  test('sample from posterior', () => {
    const xTrain = tf.tensor2d([[0], [1], [2]])
    const yTrain = tf.tensor1d([0, 1, 0])
    const xTest = tf.tensor2d([[0.5], [1.5]])

    const gprm = new GaussianProcessRegressionModel({
      kernel: new SquaredExponential(),
      indexPoints: xTrain,
      observations: yTrain,
      observationNoiseVariance: 0.01
    })

    const s = gprm.sample(xTest)
    expect(s.shape).toEqual([2])

    const sMulti = gprm.sample(xTest, [5])
    expect(sMulti.shape).toEqual([5, 2])

    s.dispose(); sMulti.dispose()
    xTrain.dispose(); yTrain.dispose(); xTest.dispose()
  })

  test('logMarginalLikelihood is finite', () => {
    const xTrain = tf.tensor2d([[0], [1], [2]])
    const yTrain = tf.tensor1d([0, 1, 0])

    const gprm = new GaussianProcessRegressionModel({
      kernel: new SquaredExponential(),
      indexPoints: xTrain,
      observations: yTrain,
      observationNoiseVariance: 0.1
    })

    const lml = gprm.logMarginalLikelihood()
    expect(isFinite(lml.dataSync()[0])).toBe(true)
    lml.dispose()
    xTrain.dispose(); yTrain.dispose()
  })

  test('predictiveNoiseVariance increases uncertainty', () => {
    const xTrain = tf.tensor2d([[0], [1]])
    const yTrain = tf.tensor1d([0, 1])
    const xTest = tf.tensor2d([[0.5]])

    const gprmClean = new GaussianProcessRegressionModel({
      kernel: new SquaredExponential(),
      indexPoints: xTrain,
      observations: yTrain,
      observationNoiseVariance: 0.01,
      predictiveNoiseVariance: 0
    })

    const gprmNoisy = new GaussianProcessRegressionModel({
      kernel: new SquaredExponential(),
      indexPoints: xTrain,
      observations: yTrain,
      observationNoiseVariance: 0.01,
      predictiveNoiseVariance: 1
    })

    const { covariance: covClean } = gprmClean.predict(xTest)
    const { covariance: covNoisy } = gprmNoisy.predict(xTest)

    expect(covNoisy.dataSync()[0]).toBeGreaterThan(covClean.dataSync()[0])

    covClean.dispose(); covNoisy.dispose()
    xTrain.dispose(); yTrain.dispose(); xTest.dispose()
  })
})
