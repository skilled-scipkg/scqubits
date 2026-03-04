---
name: scqubits-index
description: Router skill for scqubits. Start docs-first, then dispatch to build/circuit workflows or advanced simulation workflows with tests/source escalation only when needed.
---

# scqubits Skills Index

## Route the request
- Installation, environment setup, custom `Circuit`, hierarchical diagonalization, `HilbertSpace` composition, interaction parsing, or diagonalizer selection -> `../scqubits-build-and-install/SKILL.md`.
- Advanced model simulation (`Transmon`/`Fluxonium`/etc.), parameter sweeps, coherence/noise analysis, serialization/reproducibility, or multiprocessing tuning -> `../scqubits-advanced-simulation/SKILL.md`.
- Broad "where is this documented?" questions -> start with local `README.md` plus external docs links, then route to one topic skill.

## Topic skills
- `scqubits-build-and-install`: install/setup, custom `Circuit`, hierarchical configuration, composite Hilbert-space interactions, diagonalizer troubleshooting.
- `scqubits-advanced-simulation`: single-qubit and composite simulation workflows, `ParameterSweep`, coherence/noise channels, data persistence, parallel execution checks.

## Documentation-first inputs
- Local fallback doc: `README.md`
- User docs: `https://scqubits.readthedocs.io`
- Docs source: `https://github.com/scqubits/scqubits-doc`
- Example notebooks: `https://github.com/scqubits/scqubits-examples`

## Escalation sequence
1. Open the selected topic `SKILL.md`.
2. If docs detail is missing, open that topic’s doc map file:
   - `../scqubits-build-and-install/references/doc_map.md`
   - `../scqubits-advanced-simulation/references/doc_map.md`
3. Use tests for behavior-level truth before source dives.
4. If still unresolved, open that topic’s source map file:
   - `../scqubits-build-and-install/references/source_map.md`
   - `../scqubits-advanced-simulation/references/source_map.md`
5. Run targeted search (for example: `rg -n "<symbol_or_keyword>" scqubits scqubits/tests`).

## Shared behavior-test roots
- `scqubits/tests/test_circuit.py`
- `scqubits/tests/test_hilbertspace.py`
- `scqubits/tests/test_diag.py`
- `scqubits/tests/test_parametersweep.py`
- `scqubits/tests/test_noise.py`

## Source roots
- `scqubits/core`
- `scqubits/io_utils`
- `scqubits/utils`
