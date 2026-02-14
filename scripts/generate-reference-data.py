#!/usr/bin/env python3
"""
Generate reference test data for stoch distributions using scipy.stats.

Outputs JSON files to tests/reference-data/<dist>.json.
All values are cast to float32 to match tf.js default dtype.

Usage:
  python3 scripts/generate-reference-data.py
"""

import json
import os
import numpy as np
from scipy import stats

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'reference-data')
SEED = 42
N_SAMPLES = 100000


def safe_float(x):
  """Handle inf/nan for JSON serialization, cast to float32."""
  x = float(x)
  if np.isnan(x):
    return None
  if np.isinf(x):
    return 1e38 if x > 0 else -1e38
  return float(np.float32(x))


def to_f32(x):
  """Cast to float32 for tf.js compatibility, handling inf/nan."""
  if isinstance(x, (list, np.ndarray)):
    return [safe_float(v) for v in x]
  if isinstance(x, (int, float, np.floating, np.integer)):
    return safe_float(x)
  return x


def generate_normal():
  test_cases = []
  param_sets = [
    {'loc': 0.0, 'scale': 1.0},
    {'loc': 5.0, 'scale': 0.5},
    {'loc': -2.0, 'scale': 3.0},
    {'loc': 0.0, 'scale': 0.01},
    {'loc': 100.0, 'scale': 10.0},
  ]
  points = [-3.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0]

  for params in param_sets:
    dist = stats.norm(loc=params['loc'], scale=params['scale'])
    rng = np.random.RandomState(SEED)
    samples = dist.rvs(size=N_SAMPLES, random_state=rng).astype(np.float32)

    test_cases.append({
      'params': params,
      'points': to_f32(points),
      'expected': {
        'log_prob': to_f32([dist.logpdf(x) for x in points]),
        'prob': to_f32([dist.pdf(x) for x in points]),
        'cdf': to_f32([dist.cdf(x) for x in points]),
        'mean': safe_float(dist.mean()),
        'variance': safe_float(dist.var()),
        'stddev': safe_float(dist.std()),
        'entropy': safe_float(dist.entropy()),
      },
      'sample_stats': {
        'n': N_SAMPLES,
        'empirical_mean': to_f32(float(np.mean(samples))),
        'empirical_variance': to_f32(float(np.var(samples, ddof=1))),
      }
    })

  kl_pairs = [
    ({'loc': 0.0, 'scale': 1.0}, {'loc': 1.0, 'scale': 2.0}),
    ({'loc': 0.0, 'scale': 1.0}, {'loc': 0.0, 'scale': 1.0}),
    ({'loc': -1.0, 'scale': 0.5}, {'loc': 2.0, 'scale': 3.0}),
  ]

  kl_divergences = []
  for p_params, q_params in kl_pairs:
    p = stats.norm(loc=p_params['loc'], scale=p_params['scale'])
    q = stats.norm(loc=q_params['loc'], scale=q_params['scale'])
    # KL(p||q) for normal: log(sq/sp) + (sp^2 + (mp-mq)^2)/(2*sq^2) - 1/2
    sp, sq = p_params['scale'], q_params['scale']
    mp, mq = p_params['loc'], q_params['loc']
    kl = np.log(sq / sp) + (sp**2 + (mp - mq)**2) / (2 * sq**2) - 0.5
    kl_divergences.append({
      'p': p_params,
      'q': q_params,
      'expected': to_f32(kl)
    })

  return {
    'distribution': 'Normal',
    'test_cases': test_cases,
    'kl_divergence': kl_divergences
  }


def generate_bernoulli():
  test_cases = []
  param_sets = [
    {'probs': 0.5},
    {'probs': 0.1},
    {'probs': 0.9},
    {'probs': 0.01},
    {'probs': 0.99},
  ]
  points = [0.0, 1.0]

  for params in param_sets:
    p = params['probs']
    dist = stats.bernoulli(p)

    test_cases.append({
      'params': params,
      'points': points,
      'expected': {
        'log_prob': to_f32([dist.logpmf(x) for x in points]),
        'prob': to_f32([dist.pmf(x) for x in points]),
        'cdf': to_f32([dist.cdf(x) for x in points]),
        'mean': safe_float(dist.mean()),
        'variance': safe_float(dist.var()),
        'entropy': safe_float(dist.entropy()),
      }
    })

  return {
    'distribution': 'Bernoulli',
    'test_cases': test_cases,
    'kl_divergence': []
  }


def generate_uniform():
  test_cases = []
  param_sets = [
    {'low': 0.0, 'high': 1.0},
    {'low': -5.0, 'high': 5.0},
    {'low': 2.0, 'high': 2.5},
    {'low': 0.0, 'high': 100.0},
  ]

  for params in param_sets:
    dist = stats.uniform(loc=params['low'], scale=params['high'] - params['low'])
    lo, hi = params['low'], params['high']
    mid = (lo + hi) / 2
    points = [lo - 1, lo, lo + (hi - lo) * 0.25, mid, lo + (hi - lo) * 0.75, hi, hi + 1]

    test_cases.append({
      'params': params,
      'points': to_f32(points),
      'expected': {
        'log_prob': to_f32([dist.logpdf(x) for x in points]),
        'prob': to_f32([dist.pdf(x) for x in points]),
        'cdf': to_f32([dist.cdf(x) for x in points]),
        'mean': safe_float(dist.mean()),
        'variance': safe_float(dist.var()),
        'entropy': safe_float(dist.entropy()),
      }
    })

  return {
    'distribution': 'Uniform',
    'test_cases': test_cases,
    'kl_divergence': []
  }


def generate_beta():
  test_cases = []
  param_sets = [
    {'concentration1': 1.0, 'concentration0': 1.0},
    {'concentration1': 2.0, 'concentration0': 5.0},
    {'concentration1': 0.5, 'concentration0': 0.5},
    {'concentration1': 5.0, 'concentration0': 1.0},
    {'concentration1': 10.0, 'concentration0': 10.0},
  ]
  points = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

  for params in param_sets:
    a, b = params['concentration1'], params['concentration0']
    dist = stats.beta(a, b)

    test_cases.append({
      'params': params,
      'points': to_f32(points),
      'expected': {
        'log_prob': to_f32([dist.logpdf(x) for x in points]),
        'prob': to_f32([dist.pdf(x) for x in points]),
        'cdf': to_f32([dist.cdf(x) for x in points]),
        'mean': safe_float(dist.mean()),
        'variance': safe_float(dist.var()),
        'entropy': safe_float(dist.entropy()),
      }
    })

  return {
    'distribution': 'Beta',
    'test_cases': test_cases,
    'kl_divergence': []
  }


def generate_gamma():
  test_cases = []
  param_sets = [
    {'concentration': 1.0, 'rate': 1.0},
    {'concentration': 2.0, 'rate': 0.5},
    {'concentration': 0.5, 'rate': 1.0},
    {'concentration': 5.0, 'rate': 2.0},
    {'concentration': 10.0, 'rate': 10.0},
  ]
  points = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

  for params in param_sets:
    a, rate = params['concentration'], params['rate']
    dist = stats.gamma(a, scale=1.0 / rate)

    test_cases.append({
      'params': params,
      'points': to_f32(points),
      'expected': {
        'log_prob': to_f32([dist.logpdf(x) for x in points]),
        'prob': to_f32([dist.pdf(x) for x in points]),
        'cdf': to_f32([dist.cdf(x) for x in points]),
        'mean': safe_float(dist.mean()),
        'variance': safe_float(dist.var()),
        'entropy': safe_float(dist.entropy()),
      }
    })

  return {
    'distribution': 'Gamma',
    'test_cases': test_cases,
    'kl_divergence': []
  }


def generate_exponential():
  test_cases = []
  param_sets = [
    {'rate': 1.0},
    {'rate': 0.5},
    {'rate': 2.0},
    {'rate': 10.0},
    {'rate': 0.1},
  ]
  points = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

  for params in param_sets:
    rate = params['rate']
    dist = stats.expon(scale=1.0 / rate)

    test_cases.append({
      'params': params,
      'points': to_f32(points),
      'expected': {
        'log_prob': to_f32([dist.logpdf(x) for x in points]),
        'prob': to_f32([dist.pdf(x) for x in points]),
        'cdf': to_f32([dist.cdf(x) for x in points]),
        'mean': safe_float(dist.mean()),
        'variance': safe_float(dist.var()),
        'entropy': safe_float(dist.entropy()),
      }
    })

  return {
    'distribution': 'Exponential',
    'test_cases': test_cases,
    'kl_divergence': []
  }


def generate_poisson():
  test_cases = []
  param_sets = [
    {'rate': 1.0},
    {'rate': 5.0},
    {'rate': 0.5},
    {'rate': 10.0},
    {'rate': 100.0},
  ]

  for params in param_sets:
    lam = params['rate']
    dist = stats.poisson(lam)
    points = [0, 1, 2, 3, 5, 10, 20]

    test_cases.append({
      'params': params,
      'points': points,
      'expected': {
        'log_prob': to_f32([dist.logpmf(x) for x in points]),
        'prob': to_f32([dist.pmf(x) for x in points]),
        'cdf': to_f32([dist.cdf(x) for x in points]),
        'mean': safe_float(dist.mean()),
        'variance': safe_float(dist.var()),
      }
    })

  return {
    'distribution': 'Poisson',
    'test_cases': test_cases,
    'kl_divergence': []
  }


def generate_categorical():
  test_cases = []
  param_sets = [
    {'probs': [0.25, 0.25, 0.25, 0.25]},
    {'probs': [0.1, 0.2, 0.3, 0.4]},
    {'probs': [0.9, 0.05, 0.04, 0.01]},
    {'probs': [0.5, 0.5]},
  ]

  for params in param_sets:
    probs = np.array(params['probs'])
    k = len(probs)
    points = list(range(k))

    log_probs = [float(np.log(probs[i])) for i in points]
    mean_val = float(np.sum(np.arange(k) * probs))
    var_val = float(np.sum(probs * (np.arange(k) - mean_val) ** 2))
    entropy_val = float(-np.sum(probs * np.log(probs)))

    test_cases.append({
      'params': {'probs': to_f32(probs.tolist())},
      'points': points,
      'expected': {
        'log_prob': to_f32(log_probs),
        'prob': to_f32(probs[points].tolist()),
        'mean': safe_float(mean_val),
        'variance': safe_float(var_val),
        'entropy': safe_float(entropy_val),
      }
    })

  return {
    'distribution': 'Categorical',
    'test_cases': test_cases,
    'kl_divergence': []
  }


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  generators = {
    'normal': generate_normal,
    'bernoulli': generate_bernoulli,
    'uniform': generate_uniform,
    'beta': generate_beta,
    'gamma': generate_gamma,
    'exponential': generate_exponential,
    'poisson': generate_poisson,
    'categorical': generate_categorical,
  }

  for name, gen_fn in generators.items():
    data = gen_fn()
    filepath = os.path.join(OUTPUT_DIR, f'{name}.json')
    with open(filepath, 'w') as f:
      json.dump(data, f, indent=2)
    print(f'Generated {filepath} ({len(data["test_cases"])} test cases)')


if __name__ == '__main__':
  main()
