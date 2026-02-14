/**
 * Benchmark runner for stoch vs WebPPL
 *
 * Compares vectorized stoch operations against WebPPL's scalar loop approach.
 * WebPPL requires: cd reference/webppl && npm install --ignore-scripts && node scripts/adify
 *
 * Usage:
 *   npm run bench                  # CPU backend (pure JS)
 *   npm run bench:gpu              # GPU backend (CUDA via tfjs-node-gpu)
 *   npm run bench:native           # native CPU backend (tfjs-node C++ bindings)
 *   node benchmarks/run.js         # same as bench
 *   node benchmarks/run.js --gpu   # same as bench:gpu
 *   node benchmarks/run.js --native # same as bench:native
 */

const args = process.argv.slice(2)
const useGpu = args.includes('--gpu')
const useNative = args.includes('--native')

// Load the appropriate tf.js backend BEFORE anything else
let tf
if (useGpu) {
  tf = require('@tensorflow/tfjs-node-gpu')
} else if (useNative) {
  tf = require('@tensorflow/tfjs-node')
} else {
  tf = require('@tensorflow/tfjs')
}

const Benchmark = require('benchmark')

// stoch distributions (require CJS build)
let stochDists
try {
  stochDists = require('../dist/stoch.cjs.js').distributions
} catch (e) {
  console.error('Build stoch first: npm run build-dev')
  process.exit(1)
}

// Load WebPPL samplers (require adify step)
let webppl = null
try {
  const gaussian = require('../reference/webppl/src/dists/gaussian')
  const gamma = require('../reference/webppl/src/dists/gamma')
  const beta = require('../reference/webppl/src/dists/beta')
  const exponential = require('../reference/webppl/src/dists/exponential')
  const poisson = require('../reference/webppl/src/dists/poisson')
  webppl = { gaussian, gamma, beta, exponential, poisson }
  console.log('WebPPL loaded successfully.\n')
} catch (e) {
  console.log('WebPPL not available: ' + e.message.split('\n')[0])
  console.log('To enable: cd reference/webppl && npm install --ignore-scripts && node scripts/adify\n')
}

const SAMPLE_SIZES = [10000, 100000, 1000000]
const LOGPROB_SIZE = 100000

function formatNum(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(0) + 'M'
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'K'
  return String(n)
}

function runSuite(suite) {
  return new Promise((resolve) => {
    const results = []
    suite
      .on('cycle', (event) => {
        const b = event.target
        results.push({ name: b.name, hz: b.hz, mean: b.stats.mean })
        console.log('  ' + String(b))
      })
      .on('complete', () => {
        resolve(results)
      })
      .run({ async: false })
  })
}

function speedup(results) {
  if (results.length < 2) return
  const stochHz = results[0].hz
  const webpplHz = results[1].hz
  const ratio = stochHz / webpplHz
  if (ratio >= 1) {
    console.log(`  => stoch is ${ratio.toFixed(1)}x faster`)
  } else {
    console.log(`  => WebPPL is ${(1 / ratio).toFixed(1)}x faster`)
  }
}

// ---- Sampling benchmarks ----

async function benchSampling() {
  console.log('=== Sampling Benchmarks ===\n')
  const allResults = []

  // Normal
  for (const n of SAMPLE_SIZES) {
    console.log(`--- Normal sampling (n=${formatNum(n)}) ---`)
    const suite = new Benchmark.Suite()

    suite.add(`stoch Normal.sample([${formatNum(n)}])`, () => {
      const dist = new stochDists.Normal({ loc: 0, scale: 1 })
      const s = dist.sample([n])
      s.dispose()
      dist.dispose()
    })

    if (webppl) {
      suite.add(`WebPPL gaussian.sample x${formatNum(n)}`, () => {
        const arr = new Float64Array(n)
        for (let i = 0; i < n; i++) arr[i] = webppl.gaussian.sample(0, 1)
      })
    }

    const r = await runSuite(suite)
    speedup(r)
    allResults.push({ dist: 'Normal', op: 'sample', n, results: r })
    console.log()
  }

  // Gamma
  for (const n of SAMPLE_SIZES) {
    console.log(`--- Gamma sampling (n=${formatNum(n)}) ---`)
    const suite = new Benchmark.Suite()

    suite.add(`stoch Gamma.sample([${formatNum(n)}])`, () => {
      const dist = new stochDists.Gamma({ concentration: 2, rate: 1 })
      const s = dist.sample([n])
      s.dispose()
      dist.dispose()
    })

    if (webppl) {
      // WebPPL Gamma uses (shape, scale) not (concentration, rate)
      // rate=1 means scale=1/rate=1
      suite.add(`WebPPL gamma.sample x${formatNum(n)}`, () => {
        const arr = new Float64Array(n)
        for (let i = 0; i < n; i++) arr[i] = webppl.gamma.sample(2, 1)
      })
    }

    const r = await runSuite(suite)
    speedup(r)
    allResults.push({ dist: 'Gamma', op: 'sample', n, results: r })
    console.log()
  }

  // Beta
  for (const n of SAMPLE_SIZES) {
    console.log(`--- Beta sampling (n=${formatNum(n)}) ---`)
    const suite = new Benchmark.Suite()

    suite.add(`stoch Beta.sample([${formatNum(n)}])`, () => {
      const dist = new stochDists.Beta({ concentration1: 2, concentration0: 5 })
      const s = dist.sample([n])
      s.dispose()
      dist.dispose()
    })

    if (webppl) {
      suite.add(`WebPPL beta.sample x${formatNum(n)}`, () => {
        const arr = new Float64Array(n)
        for (let i = 0; i < n; i++) arr[i] = webppl.beta.sample(2, 5)
      })
    }

    const r = await runSuite(suite)
    speedup(r)
    allResults.push({ dist: 'Beta', op: 'sample', n, results: r })
    console.log()
  }

  // Exponential
  for (const n of SAMPLE_SIZES) {
    console.log(`--- Exponential sampling (n=${formatNum(n)}) ---`)
    const suite = new Benchmark.Suite()

    suite.add(`stoch Exponential.sample([${formatNum(n)}])`, () => {
      const dist = new stochDists.Exponential({ rate: 2 })
      const s = dist.sample([n])
      s.dispose()
      dist.dispose()
    })

    if (webppl) {
      suite.add(`WebPPL Exponential.sample x${formatNum(n)}`, () => {
        const d = new webppl.exponential.Exponential({ a: 2 })
        const arr = new Float64Array(n)
        for (let i = 0; i < n; i++) arr[i] = d.sample()
      })
    }

    const r = await runSuite(suite)
    speedup(r)
    allResults.push({ dist: 'Exponential', op: 'sample', n, results: r })
    console.log()
  }

  return allResults
}

// ---- Log-prob benchmarks ----

async function benchLogProb() {
  console.log('=== Log-Probability Benchmarks ===\n')
  const allResults = []
  const n = LOGPROB_SIZE

  // Normal logProb
  console.log(`--- Normal logProb (n=${formatNum(n)}) ---`)
  {
    const suite = new Benchmark.Suite()
    const points = tf.randomUniform([n], -5, 5)
    const pointsArr = Array.from(points.dataSync())

    suite.add(`stoch Normal.logProb(${formatNum(n)} points)`, () => {
      const dist = new stochDists.Normal({ loc: 0, scale: 1 })
      const lp = dist.logProb(points)
      lp.dispose()
      dist.dispose()
    })

    if (webppl) {
      suite.add(`WebPPL gaussian.score x${formatNum(n)}`, () => {
        let sum = 0
        for (let i = 0; i < n; i++) sum += webppl.gaussian.score(0, 1, pointsArr[i])
      })
    }

    const r = await runSuite(suite)
    speedup(r)
    allResults.push({ dist: 'Normal', op: 'logProb', n, results: r })
    points.dispose()
    console.log()
  }

  // Gamma logProb
  console.log(`--- Gamma logProb (n=${formatNum(n)}) ---`)
  {
    const suite = new Benchmark.Suite()
    const points = tf.randomUniform([n], 0.01, 10)
    const pointsArr = Array.from(points.dataSync())

    suite.add(`stoch Gamma.logProb(${formatNum(n)} points)`, () => {
      const dist = new stochDists.Gamma({ concentration: 2, rate: 1 })
      const lp = dist.logProb(points)
      lp.dispose()
      dist.dispose()
    })

    if (webppl) {
      const GammaDist = webppl.gamma.Gamma
      suite.add(`WebPPL Gamma.score x${formatNum(n)}`, () => {
        const d = new GammaDist({ shape: 2, scale: 1 })
        let sum = 0
        for (let i = 0; i < n; i++) sum += d.score(pointsArr[i])
      })
    }

    const r = await runSuite(suite)
    speedup(r)
    allResults.push({ dist: 'Gamma', op: 'logProb', n, results: r })
    points.dispose()
    console.log()
  }

  // Beta logProb
  console.log(`--- Beta logProb (n=${formatNum(n)}) ---`)
  {
    const suite = new Benchmark.Suite()
    const points = tf.randomUniform([n], 0.01, 0.99)
    const pointsArr = Array.from(points.dataSync())

    suite.add(`stoch Beta.logProb(${formatNum(n)} points)`, () => {
      const dist = new stochDists.Beta({ concentration1: 2, concentration0: 5 })
      const lp = dist.logProb(points)
      lp.dispose()
      dist.dispose()
    })

    if (webppl) {
      const BetaDist = webppl.beta.Beta
      suite.add(`WebPPL Beta.score x${formatNum(n)}`, () => {
        const d = new BetaDist({ a: 2, b: 5 })
        let sum = 0
        for (let i = 0; i < n; i++) sum += d.score(pointsArr[i])
      })
    }

    const r = await runSuite(suite)
    speedup(r)
    allResults.push({ dist: 'Beta', op: 'logProb', n, results: r })
    points.dispose()
    console.log()
  }

  return allResults
}

// ---- Batched operations (stoch advantage) ----

async function benchBatched() {
  console.log('=== Batched Operations (stoch only) ===\n')
  console.log('WebPPL has no batching support — these show stoch unique capability.\n')

  const batchSizes = [10, 100, 1000]
  for (const batchSize of batchSizes) {
    console.log(`--- Batched Normal (batch=${batchSize}, samples=1000) ---`)
    const suite = new Benchmark.Suite()
    const locs = Array.from({ length: batchSize }, (_, i) => i * 0.1)

    suite.add(`stoch Normal(batch=${batchSize}).sample([1000])`, () => {
      const dist = new stochDists.Normal({ loc: locs, scale: 1 })
      const s = dist.sample([1000])
      s.dispose()
      dist.dispose()
    })

    await runSuite(suite)
    console.log()
  }
}

// ---- Memory test ----

async function benchMemory() {
  console.log('=== Memory Leak Test ===\n')

  const before = tf.memory()
  console.log(`  Baseline: ${before.numTensors} tensors, ${before.numBytes} bytes`)

  for (let i = 0; i < 100; i++) {
    const dist = new stochDists.Normal({ loc: i, scale: 1 })
    const s = dist.sample([1000])
    const lp = dist.logProb(0.5)
    s.dispose()
    lp.dispose()
    dist.dispose()
  }

  const after = tf.memory()
  console.log(`  After 100 create/sample/dispose cycles: ${after.numTensors} tensors, ${after.numBytes} bytes`)
  const leaked = after.numTensors - before.numTensors
  console.log(`  Leak: ${leaked} tensors, ${after.numBytes - before.numBytes} bytes`)
  console.log(`  Status: ${leaked === 0 ? 'PASS — no leaks' : 'FAIL — tensor leak detected'}\n`)
}

// ---- Summary table ----

function printSummary(samplingResults, logProbResults) {
  if (!webppl) return

  console.log('=== Summary ===\n')
  console.log('| Operation | n | stoch (ops/s) | WebPPL (ops/s) | Speedup |')
  console.log('|---|---|---|---|---|')

  const all = [...samplingResults, ...logProbResults]
  for (const { dist, op, n, results } of all) {
    if (results.length < 2) continue
    const stochHz = results[0].hz
    const webpplHz = results[1].hz
    const ratio = stochHz / webpplHz
    const arrow = ratio >= 1 ? `**${ratio.toFixed(1)}x**` : `${(1 / ratio).toFixed(1)}x slower`
    console.log(`| ${dist}.${op} | ${formatNum(n)} | ${stochHz.toFixed(0)} | ${webpplHz.toFixed(0)} | ${arrow} |`)
  }
  console.log()
}

// ---- Main ----

async function main() {
  console.log('stoch Benchmarks')
  console.log('=================\n')
  console.log(`TensorFlow.js backend: ${tf.getBackend() || 'cpu'}`)
  console.log(`Platform: ${process.platform} ${process.arch}`)
  console.log(`Node.js: ${process.version}\n`)

  await benchMemory()
  const samplingResults = await benchSampling()
  const logProbResults = await benchLogProb()
  await benchBatched()
  printSummary(samplingResults, logProbResults)

  console.log('Done.')
}

main().catch(console.error)
