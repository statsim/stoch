"""
Generate reference data for Cholesky decomposition + gradients from PyTorch.
Used to validate the stoch implementation against a known-correct autograd.
"""
import torch
import json
import numpy as np

torch.set_default_dtype(torch.float64)  # high precision reference

def cholesky_reference(A_list, loss_type='sum'):
    """Compute L, gradient dA for a given loss through cholesky."""
    A = torch.tensor(A_list, dtype=torch.float64, requires_grad=True)
    L = torch.linalg.cholesky(A)

    if loss_type == 'sum':
        loss = L.sum()
    elif loss_type == 'sum_sq':
        loss = (L ** 2).sum()
    elif loss_type == 'log_diag':
        # sum of log of diagonal elements of L
        loss = torch.log(torch.diag(L)).sum()
    elif loss_type == 'frobenius':
        loss = torch.norm(L, p='fro')
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')

    loss.backward()

    return {
        'L': L.detach().numpy().tolist(),
        'loss': loss.item(),
        'grad': A.grad.numpy().tolist(),
        'loss_type': loss_type
    }


def triangular_solve_reference(L_list, B_list, lower=True, adjoint=False):
    """Compute X = solve(L, B) or solve(L^T, B)."""
    L = torch.tensor(L_list, dtype=torch.float64)
    B = torch.tensor(B_list, dtype=torch.float64)

    # torch.linalg.solve_triangular
    X = torch.linalg.solve_triangular(
        L.T if adjoint else L,
        B,
        upper=adjoint if lower else not adjoint
    )
    return X.numpy().tolist()


def to_json_safe(obj):
    """Convert numpy/torch types to JSON-safe Python types."""
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, list):
        return [to_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    return obj


def main():
    results = {
        'generator': 'PyTorch ' + torch.__version__,
        'dtype': 'float64',
        'cholesky_forward': [],
        'cholesky_gradients': [],
        'triangular_solve': [],
    }

    # ─── Forward reference values ────────────────────────────────
    forward_cases = [
        {'name': '1x1', 'A': [[4.0]]},
        {'name': '2x2', 'A': [[4.0, 2.0], [2.0, 3.0]]},
        {'name': '3x3_classic', 'A': [[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]]},
        {'name': '3x3_general', 'A': [[4.0, 2.0, 0.5], [2.0, 5.0, 1.0], [0.5, 1.0, 3.0]]},
        {'name': '4x4_identity', 'A': [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
        {'name': '2x2_small_diag', 'A': [[1e-6, 0.0], [0.0, 1e-6]]},
        {'name': '2x2_diagonal', 'A': [[4.0, 0.0], [0.0, 9.0]]},
    ]

    # Random well-conditioned 5x5 PD matrix
    np.random.seed(42)
    X5 = np.random.randn(5, 5)
    A5 = (X5 @ X5.T + 0.1 * np.eye(5)).tolist()
    forward_cases.append({'name': '5x5_random', 'A': A5})

    # Random 10x10
    X10 = np.random.randn(10, 10)
    A10 = (X10 @ X10.T + 0.1 * np.eye(10)).tolist()
    forward_cases.append({'name': '10x10_random', 'A': A10})

    for case in forward_cases:
        A = torch.tensor(case['A'], dtype=torch.float64)
        L = torch.linalg.cholesky(A)
        # Verify roundtrip
        reconstructed = L @ L.T
        max_err = (A - reconstructed).abs().max().item()
        results['cholesky_forward'].append({
            'name': case['name'],
            'A': case['A'],
            'L': L.numpy().tolist(),
            'roundtrip_max_error': max_err,
        })

    # ─── Gradient reference values ───────────────────────────────
    grad_cases = [
        # 2x2, multiple loss types
        {'name': '2x2_sum', 'A': [[4.0, 2.0], [2.0, 3.0]], 'loss': 'sum'},
        {'name': '2x2_sum_sq', 'A': [[4.0, 2.0], [2.0, 3.0]], 'loss': 'sum_sq'},
        {'name': '2x2_log_diag', 'A': [[4.0, 2.0], [2.0, 3.0]], 'loss': 'log_diag'},
        {'name': '2x2_frobenius', 'A': [[4.0, 2.0], [2.0, 3.0]], 'loss': 'frobenius'},
        # 3x3
        {'name': '3x3_sum', 'A': [[25.0, 15.0, -5.0], [15.0, 18.0, 0.0], [-5.0, 0.0, 11.0]], 'loss': 'sum'},
        {'name': '3x3_sum_sq', 'A': [[4.0, 2.0, 0.5], [2.0, 5.0, 1.0], [0.5, 1.0, 3.0]], 'loss': 'sum_sq'},
        {'name': '3x3_log_diag', 'A': [[4.0, 2.0, 0.5], [2.0, 5.0, 1.0], [0.5, 1.0, 3.0]], 'loss': 'log_diag'},
        # Diagonal matrices (simple analytic check)
        {'name': '2x2_diag_sum', 'A': [[4.0, 0.0], [0.0, 9.0]], 'loss': 'sum'},
        {'name': '2x2_diag_log_diag', 'A': [[4.0, 0.0], [0.0, 9.0]], 'loss': 'log_diag'},
        # 5x5 random
        {'name': '5x5_sum', 'A': A5, 'loss': 'sum'},
        {'name': '5x5_sum_sq', 'A': A5, 'loss': 'sum_sq'},
        # Near-singular with jitter (manually add jitter before computing)
        {'name': '2x2_near_singular_jittered', 'A': [[1.001, 0.999], [0.999, 1.001]], 'loss': 'sum'},
    ]

    for case in grad_cases:
        ref = cholesky_reference(case['A'], case['loss'])
        results['cholesky_gradients'].append({
            'name': case['name'],
            'A': case['A'],
            'loss_type': case['loss'],
            'L': ref['L'],
            'loss_value': ref['loss'],
            'grad': ref['grad'],
        })

    # ─── Triangular solve reference ──────────────────────────────
    solve_cases = [
        {
            'name': '2x2_lower',
            'L': [[2.0, 0.0], [1.0, 3.0]],
            'B': [[4.0], [7.0]],
            'lower': True, 'adjoint': False,
        },
        {
            'name': '3x3_lower',
            'L': [[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]],
            'B': [[1.0], [8.0], [32.0]],
            'lower': True, 'adjoint': False,
        },
        {
            'name': '2x2_lower_adjoint',
            'L': [[2.0, 0.0], [1.0, 3.0]],
            'B': [[5.0], [6.0]],
            'lower': True, 'adjoint': True,
        },
        {
            'name': '3x3_lower_adjoint',
            'L': [[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]],
            'B': [[10.0], [20.0], [18.0]],
            'lower': True, 'adjoint': True,
        },
        {
            'name': '3x3_multi_rhs',
            'L': [[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [4.0, 5.0, 6.0]],
            'B': [[1.0, 6.0], [8.0, 21.0], [32.0, 74.0]],
            'lower': True, 'adjoint': False,
        },
    ]

    for case in solve_cases:
        X = triangular_solve_reference(
            case['L'], case['B'],
            lower=case['lower'], adjoint=case['adjoint']
        )
        results['triangular_solve'].append({
            'name': case['name'],
            'L': case['L'],
            'B': case['B'],
            'lower': case['lower'],
            'adjoint': case['adjoint'],
            'X': X,
        })

    # Write JSON
    output = to_json_safe(results)
    with open('tests/reference-data/cholesky-reference.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Generated {len(results['cholesky_forward'])} forward cases")
    print(f"Generated {len(results['cholesky_gradients'])} gradient cases")
    print(f"Generated {len(results['triangular_solve'])} solve cases")

    # Print a few key gradient values for quick inspection
    for g in results['cholesky_gradients']:
        print(f"\n{g['name']} (loss={g['loss_type']}):")
        print(f"  loss = {g['loss_value']:.10f}")
        print(f"  grad = {np.array(g['grad'])}")


if __name__ == '__main__':
    main()
