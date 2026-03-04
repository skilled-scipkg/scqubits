---
name: scqubits-build-and-install
description: Use this skill for scqubits install/setup and Circuit/HilbertSpace bootstrapping, including hierarchical configuration, interaction setup, and diagonalizer troubleshooting.
---

# scqubits: Build, Install, and Composite Setup

## When to use
- Use this skill for install/setup plus startup workflows around `Circuit`, hierarchical diagonalization, `HilbertSpace`, and diagonalizer selection.
- If the request is only routing/navigation, start at `../scqubits-index/SKILL.md`.
- For advanced sweeps/noise/coherence workflows, switch to `../scqubits-advanced-simulation/SKILL.md`.

## Quick start commands
```bash
conda install -c conda-forge scqubits
python - <<'PY'
import scqubits as scq
print("scqubits version:", scq.__version__)
print("known diag methods:", len(scq.DIAG_METHODS))
PY
```

```bash
# Optional local-source development install from this repository
pip install -e .
```

## Circuit startup recipe
```python
import numpy as np
import scqubits as scq

yaml = """
branches:
- [JJ, 0, 1, 1, 15]
- [C, 1, 2, 2]
- [L, 2, 0, 0.4]
- [C, 2, 0, 0.2]
- [C, 2, 3, 0.5]
- [L, 3, 0, 0.5]
"""

circ = scq.Circuit(yaml, from_file=False, ext_basis="discretized")
circ.configure(transformation_matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 1, 1]]))
circ.cutoff_n_1, circ.cutoff_ext_2, circ.cutoff_ext_3 = 20, 10, 10
circ.configure(system_hierarchy=[[1], [2, 3]], subsystem_trunc_dims=[20, 30])
circ.ng1 = 0.5
evals = circ.eigenvals(evals_count=8)
print(evals - evals[0])
```

## Composite Hilbert-space recipe
```python
import scqubits as scq

tmon = scq.Transmon(EJ=40.0, EC=0.2, ng=0.0, ncut=40, truncated_dim=3)
res = scq.Oscillator(E_osc=6.0, truncated_dim=4)
hs = scq.HilbertSpace([tmon, res])
hs.add_interaction(g_strength=0.1, op1=tmon.n_operator, op2=res.creation_operator, add_hc=True)
evals = hs.eigenvals(evals_count=10)
hs.generate_lookup(ordering="DE")
n_dressed = hs.op_in_dressed_eigenbasis(tmon.n_operator, truncated_dim=10)
```

## Validation checkpoints
- Import check: `scqubits` imports and `scq.DIAG_METHODS` is non-empty.
- Cutoff/truncation convergence: raise `cutoff_n_*`, `cutoff_ext_*`, or `truncated_dim` and verify low transitions are stable.
- Hierarchy check: `configure(system_hierarchy=..., subsystem_trunc_dims=...)` runs without rollback exceptions.
- Interaction check: `hs.hamiltonian().isherm` is `True` for symmetric couplings (`add_hc=True`).
- Diagonalizer parity: compare custom method vs default using `np.allclose` at fixed `evals_count`.

## Fast behavior checks (pytest)
```bash
pytest -q scqubits/tests/test_circuit.py::TestCircuit::test_eigenvals_discretized
pytest -q scqubits/tests/test_hilbertspace.py::TestHilbertSpace::test_HilbertSpace_diagonalize_hamiltonian
pytest -q scqubits/tests/test_diag.py::test_custom_diagonalization_evals_method_matches_default
```

## Common failure signatures
- `Circuit instance cannot be initialized with both input_string and a symbolic hamiltonian.` -> `scqubits/core/circuit.py`.
- `Invalid choice for basis_completion: must be 'heuristic' or 'canonical'.` -> `scqubits/core/circuit.py`.
- `The truncated dimensions attribute for hierarchical diagonalization is not set.` -> `scqubits/core/circuit.py`.
- `Invalid ... evals_method/esys_method ...` -> `scqubits/core/qubit_base.py`, `scqubits/tests/test_diag.py`.
- `Noise methods are not generated... use configure() with generate_noise_methods=True` -> `scqubits/core/circuit.py`.
- `Invalid Interaction Term` / interaction parsing failures -> `scqubits/core/hilbert_space.py`.

## Escalation order
1. Start with `README.md`.
2. Use `references/doc_map.md` for the doc/test map.
3. Use behavior tests in `scqubits/tests/test_circuit.py`, `scqubits/tests/test_hilbertspace.py`, and `scqubits/tests/test_diag.py`.
4. If unresolved, use `references/source_map.md` and inspect ranked source entry points.

## Core references
- `README.md`
- `references/doc_map.md`
- `references/source_map.md`
- External docs: `https://scqubits.readthedocs.io`
- Examples: `https://github.com/scqubits/scqubits-examples`
