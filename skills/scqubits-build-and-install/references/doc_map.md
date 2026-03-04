# scqubits doc map: build-and-install topic

Generated from local docs/tests:
- `README.md`
- `scqubits/tests/test_circuit.py`
- `scqubits/tests/test_hilbertspace.py`
- `scqubits/tests/test_diag.py`

## Primary local docs
- `README.md`:
  - Install guidance: `conda install -c conda-forge scqubits` (preferred for Python 3.9-3.12).
  - `pip install scqubits` exists but README warns against pip inside conda environments.
  - Local narrative docs are sparse; behavior details come from tests and source.

## Behavior references (executable)
- `scqubits/tests/test_circuit.py`:
  - YAML/symbolic circuit construction, hierarchical setup, qutip dynamics, parameter sweeps.
- `scqubits/tests/test_hilbertspace.py`:
  - `HilbertSpace` initialization, interaction interfaces, Hermiticity, dressed-basis operator checks.
- `scqubits/tests/test_diag.py`:
  - valid/invalid `evals_method` and `esys_method` handling, parity with default diagonalization.

## Practical smoke commands
```bash
python - <<'PY'
import scqubits as scq
print(scq.__version__)
print("diag method count:", len(scq.DIAG_METHODS))
PY
```

```bash
pytest -q scqubits/tests/test_circuit.py::TestCircuit::test_eigenvals_discretized
pytest -q scqubits/tests/test_hilbertspace.py::TestHilbertSpace::test_HilbertSpace_diagonalize_hamiltonian
pytest -q scqubits/tests/test_diag.py::test_custom_diagonalization_evals_method_matches_default
```

## External documentation handoff
- `https://scqubits.readthedocs.io`
- `https://github.com/scqubits/scqubits-doc`
- `https://github.com/scqubits/scqubits-examples`

## If unresolved
- Use function-level source map: `source_map.md`.
