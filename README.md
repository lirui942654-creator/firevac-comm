# FireEvac-Comm

A Gymnasium-compatible Dec-POMDP benchmark for **communication-constrained
multi-agent fire evacuation**, with an ISO 13571 fractional effective dose
(FED) incapacitation model and a hard, measured per-step message budget.

> **Status**: research artifact. The accompanying study's central claims were
> deliberately narrow after statistical review — see `docs/FINDINGS.md` for an
> honest summary of what the experiments do and do not support.

## Why this exists

Existing fire-evacuation simulators (cellular automata, FDS+Evac, Pathfinder)
do not expose communication as a controllable, budgeted resource, and existing
communication-constrained MARL testbeds do not model survival pressure.
FireEvac-Comm couples both:

- **Fire dynamics**: cellular-automaton spread + smoke diffusion
- **Incapacitation**: FED accumulation following the ISO 13571
  fractional-dose formulation (CO-equivalent concentration-time integral,
  calibrated so that loitering in dense smoke incapacitates in ~40 steps ≈ 2 min)
- **Communication channel**: configurable radius / packet loss / send mode
  (fixed frequency, or a learnable `SEND` action)
- **Two information-structure regimes**:
  - *Regime D* (`wall`): discrete goal information (exit location must be
    gossiped from a single informed leader)
  - *Regime F* (`spreading`): continuous field information (the fire map
    itself, constantly going stale)
- **Standard Gymnasium API**; runs on CPU only

## Install

```bash
pip install -r requirements.txt
```

## Quick start

```python
from firevac_comm.env import FireEvacEnv

env = FireEvacEnv(grid=20, n_agents=6, scenario='wall',
                  comm_mode='event', comm_radius=4.0, seed=0)
obs, info = env.reset()          # obs['obs']: (n_agents, 59)
for _ in range(env.max_steps):
    act = policy(obs)            # (n_agents,) int in {0..4}; 5 = SEND (event mode)
    obs, reward, term, trunc, info = env.step(act)
    if term or trunc:
        break
print(info['metrics'])           # success / survival / avg_fed / comm_overhead ...
```

## Reproduce the experiments

Each experiment is one command; seeds are fixed and the environment is
bit-level deterministic (same seed → identical metrics, audited).

```bash
python experiments/exp_main_52seeds.py      # Regime D main table (52 seeds)
python experiments/exp_regime_f.py          # Regime F full 7-policy matrix (40 seeds)
python experiments/exp_significance.py      # Mann-Whitney + Welch + Holm correction
python experiments/exp_robustness.py        # effect size, power analysis, loss/N/exit
python experiments/exp_halflife.py          # information half-life gradient
python figures/make_figures.py              # publication figures (matplotlib)
```

Pre-computed results are in `results/` (CSV columns map 1:1 to paper tables).

## What the experiments found (honest summary)

Under matched-overhead comparison with 52 seeds, bootstrap CIs, and
Mann-Whitney tests (Holm-corrected):

1. **Token-passing (round-robin) significantly outperforms frequency-driven
   sending** (p = 0.012, Cohen's d = 0.54). This is the only statistically
   significant scheduling contrast.
2. **Recipient-driven (targeted) scheduling showed no significant advantage**
   over frequency-driven sending (p = 0.15) once the FED model was calibrated
   to ISO 13571 — an earlier apparent advantage was traced to a
   double-counting bug in the incapacitation model (documented in
   `docs/METHODS_LOG.md`).
3. **In the continuous-field regime (Regime F), all scheduling policies are
   statistically indistinguishable.** Scheduling appears to matter only when
   information outlives its transmission delay.
4. Learned schedulers (MAPPO with a learnable SEND action, ε-greedy and
   contextual bandits) are dominated by explicit rules; online learning curves
   are flat because the scheduling reward is non-stationary.

**Practical takeaway**: before optimizing *how* agents communicate, check
whether the information they would exchange outlives its transmission delay.
If not, communication scheduling is not the bottleneck.

## Repository layout

```
firevac_comm/          # the environment
experiments/           # one script per experiment, fixed seeds
results/               # pre-computed CSVs and tables
figures/               # publication figures + generation script
docs/                  # findings summary, methods log
```

## Repository

- GitHub: https://github.com/lirui942654-creator/firevac-comm
- Zenodo DOI: (see `docs/ZENODO.md` after publishing a release)

## Citation

If you use this benchmark, please cite the Zenodo DOI (see `docs/ZENODO.md`).

## License

MIT
