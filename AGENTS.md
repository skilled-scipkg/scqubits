# scqubits — AI Agent Context

## What This Package Does

scqubits is a Python library for simulating **superconducting qubits**. It provides:

- Energy spectra, wavefunctions, and matrix elements for standard qubit types
- Composite Hilbert spaces with coupled qubits and oscillators via QuTiP
- Parameter sweeps over external parameters (flux, charge offset, etc.)
- Noise and decoherence calculations (T1, Tphi)
- Custom circuit analysis from symbolic circuit descriptions
- Pluggable diagonalization backends (CPU: scipy/PRIMME, GPU: CuPy/JAX/cuQuantum)

Citation: Quantum 5, 583 (2021). Authors: Jens Koch, Peter Groszkowski.

## Architecture Overview

```
scqubits/
├── __init__.py                 # Public API exports
├── settings.py                 # Global configuration (parallelism, plotting, GPU)
├── core/
│   ├── qubit_base.py           # ABC: QuantumSystem → QubitBaseClass → QubitBaseClass1d
│   ├── transmon.py             # Transmon, TunableTransmon
│   ├── fluxonium.py            # Fluxonium
│   ├── flux_qubit.py           # FluxQubit (3-junction)
│   ├── zeropi.py               # ZeroPi
│   ├── zeropi_full.py          # FullZeroPi (with zeta coupling)
│   ├── cos2phi_qubit.py        # Cos2PhiQubit
│   ├── generic_qubit.py        # GenericQubit (2-level system)
│   ├── oscillator.py           # Oscillator, KerrOscillator
│   ├── hilbert_space.py        # HilbertSpace, InteractionTerm, InteractionTermStr
│   ├── diag.py                 # Diagonalization backends (scipy/PRIMME/CuPy/JAX/cuQuantum)
│   ├── circuit.py              # Circuit (custom circuit from symbolic description)
│   ├── symbolic_circuit.py     # SymbolicCircuit (symbolic analysis)
│   ├── param_sweep.py          # ParameterSweep (multi-dim parameter scans)
│   ├── noise.py                # Noise/decoherence calculations
│   ├── operators.py            # Quantum operator construction utilities
│   ├── storage.py              # SpectrumData, WaveFunction, DataStore
│   ├── discretization.py       # Grid1d for spatial discretization
│   ├── units.py                # Unit system (GHz/MHz/kHz/Hz)
│   ├── central_dispatch.py     # Event system (DispatchClient mixin)
│   ├── spec_lookup.py          # Dressed↔bare state mapping
│   ├── namedslots_array.py     # NamedSlotsNdarray for parameter-labeled arrays
│   └── descriptors.py          # Python descriptors for watched attributes
├── io_utils/                   # File I/O (HDF5, serialization)
├── ui/                         # GUI widgets (ipyvuetify)
├── explorer/                   # Interactive explorer widget
├── utils/                      # Plotting, spectrum utilities, CPU switching
└── tests/                      # pytest test suite
```

## Class Hierarchy

```
QuantumSystem (ABC)                 — base for all quantum systems
├── QubitBaseClass (ABC)            — adds diag method selection, plotting, matrix elements
│   ├── QubitBaseClass1d (ABC)      — 1D qubits with potential() and wavefunction()
│   │   ├── Transmon               — EJ, EC, ng, ncut (charge basis)
│   │   │   └── TunableTransmon    — adds EJmax, d, flux
│   │   └── Fluxonium              — EJ, EC, EL, flux, cutoff (harmonic osc basis)
│   ├── FluxQubit                  — 3-junction flux qubit
│   ├── ZeroPi                     — 0-π qubit (phi discretized, theta in charge basis)
│   ├── FullZeroPi                 — Zero-Pi with ζ mode
│   └── Cos2PhiQubit               — cos(2φ) qubit
├── GenericQubit                    — simple 2-level system (parameter: E)
├── Oscillator                      — harmonic oscillator (E_osc, truncated_dim)
│   └── KerrOscillator             — adds Kerr nonlinearity (K)
└── Circuit                         — custom circuit from symbolic description
```

Every `QuantumSystem` provides:
- `hamiltonian()` → returns the Hamiltonian (ndarray, sparse, or Qobj)
- `eigenvals(evals_count)` → eigenvalues
- `eigensys(evals_count)` → (eigenvalues, eigenvectors)
- `hilbertdim()` → Hilbert space dimension
- `truncated_dim` → number of kept levels
- `id_str` → unique string identifier

Qubits additionally provide:
- `evals_method` / `esys_method` — diag method selection (string name or callable)
- Matrix element methods, plotting methods

## HilbertSpace — Composite Systems

`HilbertSpace` combines subsystems (qubits + oscillators) into a tensor product space:

```python
import scqubits as scq

tmon = scq.Transmon(EJ=30.0, EC=1.2, ng=0.0, ncut=31)
osc = scq.Oscillator(E_osc=5.0, truncated_dim=5)

hilbert_space = scq.HilbertSpace([tmon, osc])
hilbert_space.add_interaction(
    g_strength=0.1,
    op1=(tmon.n_operator, tmon),
    op2=(osc.creation_operator() + osc.annihilation_operator(), osc),
)
```

Interactions can be:
- `InteractionTerm` — operator products: `g * op1 ⊗ op2`
- `InteractionTermStr` — string expressions: `"g * cos(phi1) * n2"`

The full Hamiltonian is built by tensoring subsystem Hamiltonians and adding interaction terms, using QuTiP `Qobj` for tensor products.

## Diagonalization System (diag.py)

Pluggable backends via the `DIAG_METHODS` dictionary. Method names follow the pattern `{evals|esys}_{backend}_{format}`:

### CPU Backends
| Method | Backend | Format | Returns |
|--------|---------|--------|---------|
| `evals_scipy_dense` | SciPy | dense | eigenvalues only |
| `esys_scipy_dense` | SciPy | dense | eigenvalues + eigenvectors |
| `evals_scipy_sparse` | SciPy ARPACK | sparse | eigenvalues only |
| `esys_scipy_sparse` | SciPy ARPACK | sparse | eigenvalues + eigenvectors |
| `evals_primme_sparse` | PRIMME | sparse | eigenvalues only |
| `esys_primme_sparse` | PRIMME | sparse | eigenvalues + eigenvectors |

Variants: `_SM` (smallest magnitude), `_LA_shift-inverse`, `_LM_shift-inverse`.

### GPU Backends
| Method | Backend | Format | Returns |
|--------|---------|--------|---------|
| `evals_cupy_dense` | CuPy | dense | eigenvalues only |
| `esys_cupy_dense` | CuPy | dense | eigenvalues + eigenvectors |
| `evals_cupy_sparse` | CuPy | sparse | eigenvalues only |
| `esys_cupy_sparse` | CuPy | sparse | eigenvalues + eigenvectors |
| `evals_jax_dense` | JAX | dense | eigenvalues only |
| `esys_jax_dense` | JAX | dense | eigenvalues + eigenvectors |
| `evals_cuquantum` | cuQuantum | Qobj | eigenvalues only |
| `esys_cuquantum` | cuQuantum | Qobj | eigenvalues + eigenvectors |

### cuQuantum Eigensolver
`esys_cuquantum` uses `cuquantum.densitymat.OperatorSpectrumSolver` (Krylov-based). It:
1. Converts the Hamiltonian `Qobj` to a cuQuantum `Operator` via `qutip_cuquantum.CuQobjEvo`
2. Generates random initial states as `DensePureState`
3. Uses Krylov iteration to find lowest eigenvalues
4. Returns eigenvectors as `Qobj` with `CuState` data type (for GPU-accelerated downstream ops)

Controlled by `settings.CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE`, `CUQUANTUM_MAX_BUFFER_RATIO`, `CUQUANTUM_MAX_RESTARTS`.

### Selecting a Method
```python
# Per-qubit
tmon = scq.Transmon(..., esys_method="esys_cupy_dense")

# For HilbertSpace
hilbert_space.eigenvals(evals_count=10, evals_method="evals_cuquantum")
```

## Settings (settings.py)

| Setting | Default | Purpose |
|---------|---------|---------|
| `NUM_CPUS` | `1` | Cores for parallel processing |
| `MULTIPROC` | `"pathos"` | Multiprocessing library (`"pathos"` or `"multiprocessing"`) |
| `DISPATCH_ENABLED` | `True` | Central dispatch event system |
| `AUTORUN_SWEEP` | `True` | Auto-run ParameterSweep on init |
| `STENCIL` | `7` | Derivative stencil points |
| `FUZZY_SLICING` | `False` | Value-based array slicing |
| `OVERLAP_THRESHOLD` | `0.5` | Dressed↔bare state mapping threshold |
| `SYM_INVERSION_MAX_NODES` | `3` | Symbolic capacitance matrix inversion limit |
| `CUQUANTUM_INSTALLED` | auto-detected | Whether cuQuantum + qutip-cuquantum are available |
| `CUQUANTUM_MIN_KRYLOV_BLOCK_SIZE` | `1` | Krylov eigensolver config |
| `CUQUANTUM_MAX_BUFFER_RATIO` | `5` | Krylov eigensolver config |
| `CUQUANTUM_MAX_RESTARTS` | `20` | Krylov eigensolver config |

## Event System (central_dispatch.py)

Event-driven architecture for propagating parameter changes:

- `DispatchClient` mixin — classes that emit/listen to events
- Events: `QUANTUMSYSTEM_UPDATE`, `HILBERTSPACE_UPDATE`, `PARAMETERSWEEP_UPDATE`, `GRID_UPDATE`, `INTERACTIONTERM_UPDATE`, `CIRCUIT_UPDATE`
- When a qubit parameter changes, events propagate to `HilbertSpace` → `ParameterSweep` to invalidate caches

## ParameterSweep

Multi-dimensional parameter scans:

```python
sweep = scq.ParameterSweep(
    hilbert_space,
    paramvals_by_name={"flux": np.linspace(0, 1, 101)},
    evals_count=10,
)
```

- Sweeps over external parameters, re-diagonalizes at each point
- Caches eigenvalues, eigenvectors, dressed↔bare state maps
- Supports parallel execution via `settings.NUM_CPUS`
- Uses `NamedSlotsNdarray` for parameter-labeled result arrays

## Noise Calculations

`NoisySystem` ABC (mixed into qubit classes) provides:
- `t1_effective(...)` — effective T1 from multiple channels
- `tphi_1_over_f(...)` — dephasing from 1/f noise
- Various channel-specific methods (capacitive, inductive, quasiparticle, etc.)

## Custom Circuits

`Circuit` / `SymbolicCircuit` allow defining arbitrary superconducting circuits:

```python
circuit = scq.Circuit(
    yaml_str,          # YAML string describing circuit topology
    from_file=False,
    ext_basis="discretized",
)
```

Symbolic analysis determines normal modes, then numerical diagonalization follows.

## File I/O

- `scq.write(obj, filename)` — serialize to HDF5
- `scq.read(filename)` — deserialize
- Uses a registry-based `Serializable` protocol

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `numpy` | `>=1.14.2` | Core numerics |
| `scipy` | `>=1.5` | Eigensolvers, sparse matrices |
| `qutip` | `>=4.3.1` | Quantum objects, tensor products |
| `matplotlib` | `>=3.5.1` | Plotting |
| `sympy` | any | Symbolic circuit analysis |
| `pathos` | `>=0.3.0` | Parallel processing |
| `dill` | any | Serialization (used by pathos) |
| `tqdm` | any | Progress bars |
| `cupy` | optional | GPU dense/sparse eigensolvers |
| `jax` | optional | JAX eigensolvers |
| `cuquantum` | optional | cuQuantum Krylov eigensolver |
| `qutip-cuquantum` | optional | cuQuantum QuTiP integration |
| `primme` | optional | PRIMME sparse eigensolver |

## Public API

### Core Classes
`Transmon`, `TunableTransmon`, `Fluxonium`, `FluxQubit`, `ZeroPi`, `FullZeroPi`, `Cos2PhiQubit`, `GenericQubit`, `Oscillator`, `KerrOscillator`

### Composite Systems
`HilbertSpace`, `InteractionTerm`, `InteractionTermStr`, `ParameterSweep`

### Custom Circuits
`Circuit`, `SymbolicCircuit`

### Utilities
`Grid1d`, `DataStore`, `SpectrumData`, `DIAG_METHODS`

### Functions
`read()`, `write()`, `set_units()`, `get_units()`, `to_standard_units()`, `from_standard_units()`, `about()`, `cite()`, `calc_therm_ratio()`, `identity_wrap()`, `truncation_template()`

### GUI (optional)
`GUI`, `Explorer`
