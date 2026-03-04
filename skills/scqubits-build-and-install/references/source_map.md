# scqubits source map: build-and-install topic

Source roots:
- `scqubits/core`
- `scqubits/tests`

Use this map after `doc_map.md` and behavior tests.

## Topic query tokens
- `Circuit.__init__`
- `from_yaml`
- `configure`
- `_check_truncation_indices`
- `system_hierarchy`
- `subsystem_trunc_dims`
- `HilbertSpace.add_interaction`
- `op_in_dressed_eigenbasis`
- `evals_method`
- `esys_method`
- `DIAG_METHODS`
- `generate_noise_methods`

## Fast navigation commands
```bash
rg -n "def __init__|def from_yaml|def configure|system_hierarchy|subsystem_trunc_dims|generate_noise_methods|supported_noise_channels|effective_noise_channels" scqubits/core/circuit.py
rg -n "_check_truncation_indices|_generate_subsystems|_update_interactions|def _evals_calc|def _esys_calc" scqubits/core/circuit_routines.py
rg -n "class HilbertSpace|def add_interaction|def hamiltonian|def generate_lookup|def op_in_dressed_eigenbasis" scqubits/core/hilbert_space.py
rg -n "evals_method|esys_method|Invalid .*method|DIAG_METHODS" scqubits/core/qubit_base.py scqubits/core/diag.py scqubits/tests/test_diag.py
```

## Ranked source entry points (function-level)
- `scqubits/core/circuit.py` | `Circuit.__init__`, `from_yaml`, `configure`, `_configure`, `_configure_sym_hamiltonian`, `supported_noise_channels`, `effective_noise_channels`.
- `scqubits/core/circuit_routines.py` | `_check_truncation_indices`, `_generate_subsystems`, `_update_interactions`, `_evals_calc`, `_esys_calc`.
- `scqubits/core/hilbert_space.py` | `HilbertSpace.__init__`, `add_interaction`, `hamiltonian`, `eigenvals`, `eigensys`, `generate_lookup`, `op_in_dressed_eigenbasis`.
- `scqubits/core/qubit_base.py` | `QubitBaseClass.__init__` method validation paths for `evals_method` and `esys_method`.
- `scqubits/core/diag.py` | `DIAG_METHODS` registry (available diagonalization backends and method names).
- `scqubits/core/circuit_noise.py` | generated circuit-specific noise methods after `configure(generate_noise_methods=True)`.
- `scqubits/tests/test_circuit.py` | practical behavior checks for YAML/symbolic circuits, hierarchy, sweeps, and qutip dynamics.
- `scqubits/tests/test_hilbertspace.py` | interaction equivalence checks, Hermiticity assertions, dressed-basis projection checks.
- `scqubits/tests/test_diag.py` | invalid-name/type error expectations and custom-vs-default parity checks.
- `scqubits/__init__.py` | public imports (`Circuit`, `HilbertSpace`, `ParameterSweep`, `DIAG_METHODS`).

## Common failure signatures -> inspect here
- `Circuit instance cannot be initialized with both input_string and a symbolic hamiltonian.` -> `scqubits/core/circuit.py`.
- `Invalid choice for basis_completion: must be 'heuristic' or 'canonical'.` -> `scqubits/core/circuit.py`.
- `The truncated dimensions attribute for hierarchical diagonalization is not set.` -> `scqubits/core/circuit.py`.
- `Noise methods are not generated...` -> `scqubits/core/circuit.py`.
- `Invalid ... evals_method/esys_method ...` -> `scqubits/core/qubit_base.py`, `scqubits/tests/test_diag.py`.
- `Invalid Interaction Term` -> `scqubits/core/hilbert_space.py`.
