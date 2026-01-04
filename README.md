# Master-Pilot Joint Decision Simulation

Discrete-event simulation framework for analysing hierarchical decision-making during port approach and berthing operations based on Stackelberg game theory.

## Model Description

The simulation represents master-pilot interactions as a leader-follower game where:
- **Pilot** (leader): Proposes tactical manoeuvring actions
- **Master** (follower): Evaluates and responds with accept/challenge/override decisions

Safety outcomes depend on the balance between manoeuvring demand and recovery capacity.

## Experimental Design

| Factor | Levels |
|--------|--------|
| Risk Environment | Low, Moderate, High |
| Authority Mode | Baseline, Adaptive, Proactive |
| Tug Support | Nominal, Delayed, Degraded |

Full factorial design: 27 configurations × 50 replications = 1,350 simulations

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage

Run simulation:
```bash
python simulation.py
```

Analyse results:
```bash
python analysis.py sim_results.json --out output
```

## Output Files

- `sim_results.json` - Raw simulation data
- `output/scenario_aggregate.csv` - Summary statistics
- `output/loss_*.csv` - Loss-of-control pivot tables
- `output/fig_*.png` - Publication figures

## References

- IMO Resolution A.960(23), 2004
- MAIB Annual Report, 2019
