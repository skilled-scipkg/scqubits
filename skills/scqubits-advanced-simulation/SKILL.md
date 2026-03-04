---
name: scqubits-advanced-simulation
description: Use this skill for advanced scqubits simulations: qubit spectra, matrix elements, parameter sweeps, noise/coherence calculations, persistence, and multiprocessing checks.
---

# scqubits: Advanced Simulation Workflows

## When to use
- Use this skill for simulation-heavy tasks after installation works: model setup, spectral sweeps, coherence/noise analysis, result persistence, and reproducibility checks.
- For install/setup or Circuit/HilbertSpace bootstrapping, use `../scqubits-build-and-install/SKILL.md`.

## Triage inputs
- Model class and parameters: `Transmon`, `TunableTransmon`, `Fluxonium`, `ZeroPi`, etc.
- Sweep target(s): `ng`, `flux`, `EJ`, coupled multi-parameter sweeps.
- Required outputs: `eigenvals`, `eigensys`, matrix elements, dressed indices, coherence times.
- Runtime constraints: `num_cpus`, `scq.settings.MULTIPROC`, optional custom diagonalizer method.

## Quick start: single-qubit spectrum + sweep
```python
import numpy as np
import scqubits as scq

q = scq.Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=80)
evals = q.eigenvals(evals_count=8)
spec = q.get_spectrum_vs_paramvals(
    param_name="ng",
    param_vals=np.linspace(-0.5, 0.5, 41),
    evals_count=6,
    subtract_ground=True,
    get_eigenstates=True,
    num_cpus=1,
)
matelem = q.matrixelement_table("n_operator", evals_count=6)
```

## Quick start: composite sweep with interactions
```python
import numpy as np
import scqubits as scq

scq.settings.MULTIPROC = "pathos"
t1 = scq.TunableTransmon(EJmax=40.0, EC=0.2, d=0.1, flux=0.0, ng=0.3, ncut=40, truncated_dim=3)
t2 = scq.TunableTransmon(EJmax=15.0, EC=0.15, d=0.2, flux=0.0, ng=0.0, ncut=30, truncated_dim=3)
res = scq.Oscillator(E_osc=4.5, truncated_dim=4)
hs = scq.HilbertSpace([t1, t2, res])
hs.add_interaction(g_strength=0.1, op1=t1.n_operator, op2=res.creation_operator, add_hc=True)
hs.add_interaction(g_strength=0.2, op1=t2.n_operator, op2=res.creation_operator, add_hc=True)

def update_hs(flux):
    t1.flux = flux
    t2.flux = 1.2 * flux

sweep = scq.ParameterSweep(
    hilbertspace=hs,
    paramvals_by_name={"flux": np.linspace(0.0, 2.0, 21)},
    update_hilbertspace=update_hs,
    evals_count=20,
    subsys_update_info={"flux": [t1, t2]},
    num_cpus=1,
)
sweep.generate_lookup(ordering="DE")
```

## Quick start: coherence and persistence
```python
import scqubits as scq

q = scq.Fluxonium(EJ=8.9, EC=2.5, EL=0.5, flux=0.5, cutoff=120)
channels = q.supported_noise_channels()
t1_eff = q.t1_effective(common_noise_options={"i": 1, "j": 0})
t2_eff = q.t2_effective(common_noise_options={"i": 1, "j": 0})

q.filewrite("fluxonium_model.h5")
q_copy = scq.read("fluxonium_model.h5")
```

## Validation checkpoints
- Convergence: increase `ncut`/`cutoff`/`truncated_dim` and confirm low-lying transition stability.
- Sweep correctness: confirm `spec.energy_table.shape[0] == len(param_vals)` and spot-check selected points with direct `eigenvals()`.
- Parallel parity: compare `num_cpus=1` vs `num_cpus>1` (same method/options) with `np.allclose`.
- Noise sanity: call `supported_noise_channels()` first, then pass only supported channels to `t1_effective`/`t2_effective`.
- Persistence: after `filewrite`+`read`, verify key arrays/parameters match.

## Practical smoke commands
```bash
pytest -q scqubits/tests/test_parametersweep.py::TestParameterSweep::test_ParameterSweep
pytest -q scqubits/tests/test_noise.py::TestNoise::test_Fluxonium
pytest -q scqubits/tests/test_spectrumlookup.py::TestParameterSweep::test_sweep_bare_eigenenergies
```

## Common failure signatures
- `Repeated id_str are not allowed in ParameterSweep.` -> `scqubits/core/param_sweep.py`.
- `subsys_update_info[...] are not found in the provided HilbertSpace object.` -> `scqubits/core/param_sweep.py`.
- `The noise_channels argument should be one of {str, list of str, or list of tuples}.` -> `scqubits/core/noise.py`.
- `Only t1 channels can contribute to effective t1 noise.` -> `scqubits/core/noise.py`.
- `Level indices 'i' and 'j' must be different, and i,j>=0` -> `scqubits/core/noise.py`.
- `scqubits multiprocessing mode set to 'pathos' ... cannot find pathos/dill` -> `scqubits/utils/cpu_switch.py`.

## Escalation order
1. Use `references/doc_map.md` for topic docs/tests.
2. Validate behavior with targeted `pytest` commands.
3. Use `references/source_map.md` for function-level entry points.
4. Run focused search (for example: `rg -n "<symbol>" scqubits/core scqubits/tests`).

## Core references
- `README.md`
- `references/doc_map.md`
- `references/source_map.md`
- External docs: `https://scqubits.readthedocs.io`
- Examples: `https://github.com/scqubits/scqubits-examples`
