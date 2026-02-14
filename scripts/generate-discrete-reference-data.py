#!/usr/bin/env python3
"""
Generate reference test data for discrete distributions in stoch using scipy.stats.

Outputs JSON files to tests/reference-data/<dist>.json.
All values are cast to float32 to match tf.js default dtype.

Usage:
  conda run -n prob python scripts/generate-discrete-reference-data.py
"""

import json
import os
import numpy as np
from scipy import stats

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'reference-data')


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


def generate_binomial():
  """Binomial(n, p): scipy.stats.binom(n, p)"""
  test_cases = []
  param_sets = [
    {'totalCount': 10, 'probs': 0.5},
    {'totalCount': 20, 'probs': 0.3},
    {'totalCount': 50, 'probs': 0.8},
  ]

  for params in param_sets:
    n = params['totalCount']
    p = params['probs']
    dist = stats.binom(n, p)

    # Test points spanning the support [0, n]
    points = sorted(set([0, 1, 2, n // 4, n // 2, 3 * n // 4, n - 1, n]))

    test_cases.append({
      'params': {
        'totalCount': n,
        'probs': safe_float(p),
      },
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
    'distribution': 'Binomial',
    'test_cases': test_cases,
  }


def generate_negative_binomial():
  """
  NegativeBinomial(totalCount=n, probs=p):
    scipy.stats.nbinom(n, p) gives P(X=k) = C(k+n-1, k) * p^n * (1-p)^k
    where n = number of successes (totalCount), p = success probability.
  """
  test_cases = []
  param_sets = [
    {'totalCount': 5, 'probs': 0.5},
    {'totalCount': 10, 'probs': 0.3},
    {'totalCount': 1, 'probs': 0.8},
  ]

  points = [0, 1, 2, 3, 5, 10, 15, 20]

  for params in param_sets:
    n = params['totalCount']
    p = params['probs']
    dist = stats.nbinom(n, p)

    test_cases.append({
      'params': {
        'totalCount': n,
        'probs': safe_float(p),
      },
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
    'distribution': 'NegativeBinomial',
    'test_cases': test_cases,
  }


def generate_geometric():
  """
  Geometric(probs=p): 0-indexed, P(X=k) = p * (1-p)^k for k = 0, 1, 2, ...
  scipy.stats.geom(p) is 1-indexed: P(X=k) = p * (1-p)^(k-1) for k = 1, 2, ...
  So we use geom with k+1 to get the 0-indexed version.
  """
  test_cases = []
  param_sets = [
    {'probs': 0.5},
    {'probs': 0.1},
    {'probs': 0.9},
  ]

  points = [0, 1, 2, 3, 5, 10]

  for params in param_sets:
    p = params['probs']
    # scipy.stats.geom is 1-indexed, so geom.logpmf(k+1, p) = log(p*(1-p)^k)
    dist = stats.geom(p)

    # 0-indexed log_prob: use scipy with k+1 shift
    log_probs = [dist.logpmf(k + 1) for k in points]
    probs_vals = [dist.pmf(k + 1) for k in points]
    # CDF: P(X <= k) in 0-indexed = P(Y <= k+1) in 1-indexed = geom.cdf(k+1)
    cdf_vals = [dist.cdf(k + 1) for k in points]

    # Mean of 0-indexed geometric: (1-p)/p
    mean_val = (1.0 - p) / p
    # Variance of 0-indexed geometric: (1-p)/p^2
    var_val = (1.0 - p) / (p * p)
    # Entropy: same as 1-indexed (just a shift)
    entropy_val = dist.entropy()

    test_cases.append({
      'params': {
        'probs': safe_float(p),
      },
      'points': points,
      'expected': {
        'log_prob': to_f32(log_probs),
        'prob': to_f32(probs_vals),
        'cdf': to_f32(cdf_vals),
        'mean': safe_float(mean_val),
        'variance': safe_float(var_val),
        'entropy': safe_float(entropy_val),
      }
    })

  return {
    'distribution': 'Geometric',
    'test_cases': test_cases,
  }


def generate_multinomial():
  """
  Multinomial(totalCount=n, probs):
    scipy.stats.multinomial(n, probs)
  """
  param_sets = [
    {
      'totalCount': 10,
      'probs': [0.25, 0.25, 0.25, 0.25],
      'test_points': [
        [3, 3, 2, 2],
        [10, 0, 0, 0],
        [0, 0, 0, 10],
        [2, 3, 3, 2],
        [1, 1, 1, 7],
      ],
    },
    {
      'totalCount': 5,
      'probs': [0.5, 0.3, 0.2],
      'test_points': [
        [3, 1, 1],
        [5, 0, 0],
        [0, 0, 5],
        [2, 2, 1],
        [1, 3, 1],
      ],
    },
    {
      'totalCount': 20,
      'probs': [0.1, 0.2, 0.3, 0.4],
      'test_points': [
        [2, 4, 6, 8],
        [5, 5, 5, 5],
        [0, 0, 0, 20],
        [20, 0, 0, 0],
        [1, 3, 7, 9],
      ],
    },
  ]

  test_cases = []
  for params in param_sets:
    n = params['totalCount']
    probs = np.array(params['probs'])
    dist = stats.multinomial(n, probs)
    test_points = params['test_points']

    log_probs = [dist.logpmf(x) for x in test_points]
    prob_vals = [dist.pmf(x) for x in test_points]

    # Mean: n * probs
    mean_val = (n * probs).tolist()
    # Variance (marginal): n * p_i * (1 - p_i)
    var_val = (n * probs * (1.0 - probs)).tolist()
    # Entropy
    entropy_val = dist.entropy()

    test_cases.append({
      'params': {
        'totalCount': n,
        'probs': to_f32(probs.tolist()),
      },
      'points': test_points,
      'expected': {
        'log_prob': to_f32(log_probs),
        'prob': to_f32(prob_vals),
        'mean': to_f32(mean_val),
        'variance': to_f32(var_val),
        'entropy': safe_float(entropy_val),
      }
    })

  return {
    'distribution': 'Multinomial',
    'test_cases': test_cases,
  }


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  generators = {
    'binomial': generate_binomial,
    'negativeBinomial': generate_negative_binomial,
    'geometric': generate_geometric,
    'multinomial': generate_multinomial,
  }

  for name, gen_fn in generators.items():
    data = gen_fn()
    filepath = os.path.join(OUTPUT_DIR, f'{name}.json')
    with open(filepath, 'w') as f:
      json.dump(data, f, indent=2)
    print(f'Generated {filepath} ({len(data["test_cases"])} test cases)')


if __name__ == '__main__':
  main()
