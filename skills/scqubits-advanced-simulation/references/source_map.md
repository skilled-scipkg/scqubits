# scqubits source map: advanced simulation topic

Source roots:
- `scqubits/core`
- `scqubits/io_utils`
- `scqubits/utils`
- `scqubits/tests`

Use this map after `doc_map.md` and targeted tests.

## Topic query tokens
- `get_spectrum_vs_paramvals`
- `matrixelement_table`
- `ParameterSweep`
- `set_update_func`
- `subsys_update_info`
- `supported_noise_channels`
- `t1_effective`
- `t2_effective`
- `tphi_1_over_f`
- `filewrite`
- `read`
- `MULTIPROC`
- `get_map_method`

## Fast navigation commands
```bash
rg -n "def get_spectrum_vs_paramvals|def matrixelement_table|evals_method|esys_method|def plot_evals_vs_paramvals" scqubits/core/qubit_base.py
rg -n "class ParameterSweep|class StoredSweep|def __init__|def set_update_func|def run|def add_sweep|def add_matelem_sweep|Repeated id_str|subsys_update_info" scqubits/core/param_sweep.py
rg -n "class NoisySystem|def supported_noise_channels|def effective_noise_channels|def t1_effective|def t2_effective|def tphi_1_over_f|Only t1 channels|noise_channels argument" scqubits/core/noise.py
rg -n "class Transmon|class TunableTransmon|class Fluxonium|def supported_noise_channels|def hamiltonian|def d_hamiltonian_d_" scqubits/core/transmon.py scqubits/core/fluxonium.py
rg -n "def write|def read|def serialize|def deserialize|def filewrite" scqubits/io_utils/fileio.py scqubits/io_utils/fileio_serializers.py
rg -n "MULTIPROC|NUM_CPUS|get_map_method|pathos|multiprocessing" scqubits/settings.py scqubits/utils/cpu_switch.py
```

## Ranked source entry points (function-level)
- `scqubits/core/qubit_base.py` | `matrixelement_table`, `get_spectrum_vs_paramvals`, `plot_evals_vs_paramvals`, and diagonalizer method validation.
- `scqubits/core/param_sweep.py` | `ParameterSweep.__init__`, `set_update_func`, `run`, `add_sweep`, `add_matelem_sweep`, `StoredSweep.new_sweep`, and parameter/subsystem validation errors.
- `scqubits/core/noise.py` | `NoisySystem.supported_noise_channels`, `effective_noise_channels`, `t1_effective`, `t2_effective`, `tphi_1_over_f`, `t1`.
- `scqubits/core/transmon.py` | `Transmon`/`TunableTransmon` initialization, Hamiltonian derivatives, and supported channel definitions.
- `scqubits/core/fluxonium.py` | `Fluxonium` initialization, Hamiltonian and derivative operators, supported channels.
- `scqubits/core/hilbert_space.py` | composite-system Hamiltonians, `generate_lookup`, dressed-basis operator projection, and interaction handling.
- `scqubits/io_utils/fileio.py` | top-level `write(...)` and `read(...)` persistence entry points.
- `scqubits/io_utils/fileio_serializers.py` | serializable protocol and `.filewrite(...)` behavior.
- `scqubits/utils/cpu_switch.py` | multiprocessing backend selection and backend-missing error paths.
- `scqubits/tests/test_parametersweep.py` | practical sweep initialization, lookup generation, and file IO behavior checks.
- `scqubits/tests/test_noise.py` | model-specific coherence channel sanity checks.
- `scqubits/tests/test_spectrumlookup.py` | lookup mapping correctness in `HilbertSpace` and `ParameterSweep`.

## Common failure signatures -> inspect here
- `Repeated id_str are not allowed in ParameterSweep.` -> `scqubits/core/param_sweep.py`.
- `Subsystems specified in subsys_update_info[...] are not found ...` -> `scqubits/core/param_sweep.py`.
- `The noise_channels argument should be one of {str, list of str, or list of tuples}.` -> `scqubits/core/noise.py`.
- `Only t1 channels can contribute to effective t1 noise.` -> `scqubits/core/noise.py`.
- `Level indices 'i' and 'j' must be different, and i,j>=0` -> `scqubits/core/noise.py`.
- `scqubits multiprocessing mode set to 'pathos' ... cannot find 'pathos'/'dill'` -> `scqubits/utils/cpu_switch.py`.
