#!/usr/bin/env python3
"""
Generate reference test data for 5 parameterized continuous distributions.

Outputs JSON files to tests/reference-data/:
  inverseGamma.json, pareto.json, weibull.json, vonMises.json, truncatedNormal.json

Format matches cauchy.json: array of param sets, each with
  params, test_points [{x, logProb, cdf}], mean, variance, entropy.

Usage:
  conda run -n prob python scripts/generate-continuous-reference-data.py
"""

import json
import os
import math
import numpy as np
from scipy import stats

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'tests', 'reference-data')


def safe_float(x):
  """Convert to Python float, replacing nan with None and clamping inf."""
  x = float(x)
  if np.isnan(x):
    return None
  if np.isinf(x):
    return 1e38 if x > 0 else -1e38
  return x


def make_entry(params, dist, test_xs):
  """Build one param-set entry in the cauchy.json format."""
  test_points = []
  for x in test_xs:
    cdf_val = float(dist.cdf(x))
    # Clamp CDF to [0, 1] to handle numerical artifacts (e.g. vonMises)
    cdf_val = max(0.0, min(1.0, cdf_val))
    test_points.append({
      'x': safe_float(x),
      'logProb': safe_float(dist.logpdf(x)),
      'cdf': safe_float(cdf_val),
    })

  return {
    'params': params,
    'test_points': test_points,
    'mean': safe_float(dist.mean()),
    'variance': safe_float(dist.var()),
    'entropy': safe_float(dist.entropy()),
  }


def generate_inverse_gamma():
  """InverseGamma(concentration, scale) via scipy.stats.invgamma(a, scale=scale)."""
  param_sets = [
    {'concentration': 2, 'scale': 1},
    {'concentration': 5, 'scale': 2},
    {'concentration': 1, 'scale': 0.5},
  ]
  test_xs = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
  entries = []
  for p in param_sets:
    dist = stats.invgamma(p['concentration'], scale=p['scale'])
    entries.append(make_entry(p, dist, test_xs))
  return entries


def generate_pareto():
  """Pareto(concentration, scale) via scipy.stats.pareto(b, scale=scale).
  concentration=b=shape, scale=x_min.
  """
  param_sets = [
    {'concentration': 2, 'scale': 1},
    {'concentration': 3, 'scale': 0.5},
    {'concentration': 1.5, 'scale': 2},
  ]
  entries = []
  for p in param_sets:
    dist = stats.pareto(p['concentration'], scale=p['scale'])
    s = p['scale']
    test_xs = [s, s * 1.5, s * 2, s * 5, s * 10, s * 20]
    entries.append(make_entry(p, dist, test_xs))
  return entries


def generate_weibull():
  """Weibull(concentration, scale) via scipy.stats.weibull_min(c, scale=scale).
  concentration=c=shape, scale=lambda.
  """
  param_sets = [
    {'concentration': 1, 'scale': 1},
    {'concentration': 2, 'scale': 2},
    {'concentration': 0.5, 'scale': 1},
  ]
  test_xs = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
  entries = []
  for p in param_sets:
    dist = stats.weibull_min(p['concentration'], scale=p['scale'])
    entries.append(make_entry(p, dist, test_xs))
  return entries


def generate_von_mises():
  """VonMises(loc, concentration) via scipy.stats.vonmises(kappa, loc=loc)."""
  param_sets = [
    {'loc': 0, 'concentration': 1},
    {'loc': math.pi / 2, 'concentration': 5},
    {'loc': 0, 'concentration': 0.1},
  ]
  test_xs = [-math.pi, -math.pi / 2, 0, math.pi / 4, math.pi / 2,
             math.pi * 3 / 4, math.pi]
  entries = []
  for p in param_sets:
    dist = stats.vonmises(p['concentration'], loc=p['loc'])
    entries.append(make_entry(p, dist, test_xs))
  return entries


def generate_truncated_normal():
  """TruncatedNormal(loc, scale, low, high) via scipy.stats.truncnorm(a, b, loc, scale).
  a = (low - loc) / scale, b = (high - loc) / scale.
  """
  param_sets = [
    {'loc': 0, 'scale': 1, 'low': -2, 'high': 2},
    {'loc': 5, 'scale': 2, 'low': 0, 'high': 10},
    {'loc': 0, 'scale': 1, 'low': 0, 'high': 1e6},
  ]
  # Test points within [low, high] for each param set
  test_points_per_set = [
    [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    [0.5, 2.0, 4.0, 5.0, 6.0, 8.0, 9.5],
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
  ]
  entries = []
  for p, test_xs in zip(param_sets, test_points_per_set):
    a = (p['low'] - p['loc']) / p['scale']
    b = (p['high'] - p['loc']) / p['scale']
    dist = stats.truncnorm(a, b, loc=p['loc'], scale=p['scale'])
    entries.append(make_entry(p, dist, test_xs))
  return entries


def write_json(filename, data):
  filepath = os.path.join(OUTPUT_DIR, filename)
  with open(filepath, 'w') as f:
    json.dump(data, f, indent=2)
  print(f'Generated {filepath} ({len(data)} param sets)')


def main():
  os.makedirs(OUTPUT_DIR, exist_ok=True)

  write_json('inverseGamma.json', generate_inverse_gamma())
  write_json('pareto.json', generate_pareto())
  write_json('weibull.json', generate_weibull())
  write_json('vonMises.json', generate_von_mises())
  write_json('truncatedNormal.json', generate_truncated_normal())

  print('\nDone. All 5 reference data files generated.')


if __name__ == '__main__':
  main()
