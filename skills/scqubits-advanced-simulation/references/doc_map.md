# scqubits doc map: advanced simulation topic

Generated from local docs/tests:
- `README.md`
- `scqubits/tests/test_parametersweep.py`
- `scqubits/tests/test_noise.py`
- `scqubits/tests/test_spectrumlookup.py`
- `scqubits/tests/test_transmon.py`
- `scqubits/tests/test_fluxonium.py`

## High-value extracted facts
- Parameter sweeps are first-class: `ParameterSweep` supports multi-parameter dictionaries plus subsystem update routing.
- Core qubit classes expose consistent spectral APIs through `eigenvals`, `eigensys`, `get_spectrum_vs_paramvals`, and matrix-element tools.
- Noise/coherence channels are model-dependent; call `supported_noise_channels()` before `t1_*`/`tphi_*` workflows.
- File persistence is built into serializable classes via `.filewrite(...)`, and objects are restored with `scqubits.read(...)`.

## Behavior references (executable)
- `scqubits/tests/test_parametersweep.py`:
  - end-to-end `ParameterSweep` construction, file IO, and lookup labeling behavior.
- `scqubits/tests/test_noise.py`:
  - channel coverage and coherence checks across multiple qubit classes.
- `scqubits/tests/test_spectrumlookup.py`:
  - `HilbertSpace.generate_lookup`, dressed/bare index mapping, and sweep lookup behavior.
- `scqubits/tests/test_transmon.py`, `scqubits/tests/test_fluxonium.py`:
  - standard spectral workflow hooks shared by common qubit models.

## Practical smoke commands
```bash
pytest -q scqubits/tests/test_parametersweep.py::TestParameterSweep::test_ParameterSweep
pytest -q scqubits/tests/test_noise.py::TestNoise::test_Transmon
pytest -q scqubits/tests/test_spectrumlookup.py::TestParameterSweep::test_sweep_bare_eigenenergies
```

## External documentation handoff
- `https://scqubits.readthedocs.io`
- `https://github.com/scqubits/scqubits-doc`
- `https://github.com/scqubits/scqubits-examples`

## If unresolved
- Use function-level source map: `source_map.md`.
