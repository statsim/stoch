import * as tf from '@tensorflow/tfjs'

/**
 * Joint distribution specified as a named dependency graph.
 *
 * Each variable is defined by a function that returns a distribution,
 * optionally conditioned on upstream variables. Supports two modes:
 *
 * Mode A — explicit deps (works under minification):
 *   { mu: { deps: [], fn: () => new Normal({ loc: 0, scale: 10 }) },
 *     x:  { deps: ['mu'], fn: ({ mu }) => new Normal({ loc: mu, scale: 1 }) } }
 *
 * Mode B — arg parsing (convenience, breaks under minification):
 *   { mu: () => new Normal({ loc: 0, scale: 10 }),
 *     x:  ({ mu }) => new Normal({ loc: mu, scale: 1 }) }
 *
 * Usage:
 *   const model = new JointDistributionNamed({ ... })
 *   const samples = model.sample()           // { mu: Tensor, x: Tensor }
 *   const lp = model.logProb(samples)        // scalar Tensor
 *   const parts = model.logProbParts(samples) // { mu: Tensor, x: Tensor }
 */
export class JointDistributionNamed {
  constructor(model, { validateArgs, name } = {}) {
    this._name = name || 'JointDistributionNamed'
    this._validateArgs = validateArgs != null ? validateArgs : false
    this._model = this._normalizeModel(model)
    this._sortedNames = this._topologicalSort()
  }

  get name() { return this._name }

  /**
   * Normalize all specs to { deps, fn } format.
   * Detects whether each entry is a plain function or { deps, fn } object.
   */
  _normalizeModel(model) {
    const normalized = {}
    for (const [name, spec] of Object.entries(model)) {
      if (typeof spec === 'function') {
        normalized[name] = { deps: this._parseDeps(spec), fn: spec }
      } else if (spec && typeof spec === 'object' && typeof spec.fn === 'function') {
        normalized[name] = { deps: spec.deps || [], fn: spec.fn }
      } else {
        throw new Error(`Invalid model spec for '${name}': expected function or { deps, fn }`)
      }
    }
    return normalized
  }

  /**
   * Parse destructured parameter names from a function's string representation.
   * Handles: ({ a, b }) => ..., function({ a, b }) { ... }
   * Returns [] for functions with no destructured params.
   */
  _parseDeps(fn) {
    const src = fn.toString()
    // Match destructured object parameter: ({ a, b, c })
    const match = src.match(/^\s*\(\s*\{([^}]*)\}\s*\)/)
      || src.match(/^\s*\(\s*\{([^}]*)\}\s*\)\s*=>/)
      || src.match(/^function\s*\w*\s*\(\s*\{([^}]*)\}\s*\)/)
      || src.match(/^\s*\{([^}]*)\}\s*=>/)
    if (!match) return []
    return match[1]
      .split(',')
      .map(s => s.trim().split(/[=:]/)[0].trim()) // handle defaults and renaming
      .filter(Boolean)
  }

  /**
   * Topological sort using Kahn's algorithm.
   * Throws on circular dependencies or unknown deps.
   */
  _topologicalSort() {
    const names = Object.keys(this._model)
    const nameSet = new Set(names)
    const inDegree = {}
    const adjacency = {} // parent -> children that depend on it

    for (const name of names) {
      inDegree[name] = 0
      adjacency[name] = []
    }

    for (const name of names) {
      for (const dep of this._model[name].deps) {
        if (!nameSet.has(dep)) {
          throw new Error(`Unknown dependency '${dep}' in '${name}'`)
        }
        adjacency[dep].push(name)
        inDegree[name]++
      }
    }

    const queue = names.filter(n => inDegree[n] === 0)
    const sorted = []

    while (queue.length > 0) {
      const node = queue.shift()
      sorted.push(node)
      for (const child of adjacency[node]) {
        inDegree[child]--
        if (inDegree[child] === 0) {
          queue.push(child)
        }
      }
    }

    if (sorted.length !== names.length) {
      throw new Error('Circular dependency detected in model')
    }

    return sorted
  }

  /**
   * Draw samples from the joint distribution.
   * Returns dict of { name: Tensor }.
   *
   * Root variables (no deps) are sampled with the requested shape.
   * Downstream variables already have sample dimensions baked into
   * their batch shape from upstream values, so we sample one per
   * batch element and squeeze the leading dimension.
   */
  sample(shape) {
    if (!shape) shape = []
    if (typeof shape === 'number') shape = [shape]
    const n = shape.reduce((a, b) => a * b, 1) || 1

    const samples = {}
    const tempDists = []

    for (const name of this._sortedNames) {
      const { deps, fn } = this._model[name]
      const depValues = {}
      for (const dep of deps) depValues[dep] = samples[dep]

      const dist = fn(depValues)
      tempDists.push(dist)
      samples[name] = this._sampleFromDist(dist, deps.length === 0 ? shape : [], deps.length === 0 ? n : 1)
    }

    // Dispose temporary distributions (their parameter tensors)
    for (const dist of tempDists) {
      if (dist.dispose) dist.dispose()
    }

    return samples
  }

  /**
   * Sample from a single component distribution.
   * Handles reshaping: squeezes leading dim for scalar sample shape,
   * reshapes to [...shape, ...batch, ...event] otherwise.
   */
  _sampleFromDist(dist, shape, n) {
    return tf.tidy(() => {
      const raw = dist._sampleN(n)
      if (n === 1 && shape.length === 0) {
        // Squeeze leading sample dim: [1, ...batch, ...event] → [...batch, ...event]
        return raw.reshape(raw.shape.slice(1))
      }
      if (shape.length > 0) {
        return raw.reshape([...shape, ...dist.batchShape, ...dist.eventShape])
      }
      return raw
    })
  }

  /**
   * Joint log probability: sum of all component log probabilities.
   * @param {Object} values - dict of { name: Tensor|number }
   * @returns {tf.Tensor} scalar
   */
  logProb(values) {
    return tf.tidy(() => {
      const parts = this._logProbPartsInternal(values)
      let total = null
      for (const name of this._sortedNames) {
        total = total === null ? parts[name] : tf.add(total, parts[name])
      }
      return total || tf.scalar(0)
    })
  }

  /**
   * Per-component log probabilities.
   * @param {Object} values - dict of { name: Tensor|number }
   * @returns {Object} dict of { name: Tensor }
   */
  logProbParts(values) {
    return this._logProbPartsInternal(values)
  }

  /**
   * Internal: compute per-component logProbs.
   * Distributions are created on-the-fly and disposed after use.
   */
  _logProbPartsInternal(values) {
    const parts = {}
    const tempDists = []

    for (const name of this._sortedNames) {
      const { deps, fn } = this._model[name]
      const depValues = {}
      for (const dep of deps) depValues[dep] = values[dep]

      const dist = fn(depValues)
      tempDists.push(dist)
      parts[name] = dist.logProb(values[name])
    }

    // Dispose temporary distributions
    for (const dist of tempDists) {
      if (dist.dispose) dist.dispose()
    }

    return parts
  }

  /**
   * Names of model variables in topological order.
   */
  get variableNames() {
    return [...this._sortedNames]
  }

  dispose() {
    // No persistent tensors — distributions are created on-the-fly
  }
}
