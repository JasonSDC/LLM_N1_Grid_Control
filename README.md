# LLM-N1 Grid Control

Emergency remedial action after **N-1 line outages** on the IEEE 39-bus system.

An LLM (or a simple rule) **only picks which generators to use**. A sequential linear program (SLP) computes **ΔP / ΔVset**. Physics checks (Check-0 / Check-1) accept or reject the action. If that fails, a load-shedding **fallback** keeps the grid safe.

---

## What it does

After a single line trip plus a load increase, the network can violate voltage or thermal limits. This repo:

1. Builds 20 reproducible N-1 + load-disturbance cases (`seed=42`)
2. Asks an intent layer to select a **small set of generators**
3. Maps that intent to continuous setpoints with **sensitivity-based SLP**
4. Verifies capacity (Check-0) and AC power flow feasibility (Check-1)
5. Falls back to voltage contraction + up to 20% load shed if needed

**Cost metric** (external, same for every method):

`J_eval = Σ|ΔP| + 100 · Σ|ΔV|`

---

## How it works

```
N-1 scenario  →  violations  →  sensitivity
        ↓
  LLM or Rule  (generator list only)
        ↓
  SLP mapping  (≤ 10 inner iterations)
        ↓
  Check-0 / Check-1
        ↓
  success → log J_eval    or    fallback
```

LLM retries with a physics hint (up to 5 rounds). Rule and OPF do not retry.

---

## Methods compared

| Method | Role |
|--------|------|
| **AC-OPF (L2)** | Post-contingency AC OPF, quadratic cost |
| **AC-OPF (L1)** | Near-L1 cost, fairer vs `J_eval` |
| **ΔV-only OPF** | Lock P, voltage setpoints only |
| **Rule+SLP** | Sensitivity top-k selection + same SLP (LLM ablation) |
| **LLM-RA** | Claude selects generators + SLP + Check-0/1 |

---

## Results (`seed=42`, 20 cases)

| | OPF (L2) | OPF (L1) | ΔV-only | Rule+SLP | LLM-RA |
|--|----------|----------|---------|----------|--------|
| Intent success (no fallback) | 17/20 | 15/20 | 8/20 | 15/20 | 14/20 |
| Overall success | **20/20** | **20/20** | **20/20** | 19/20 | 19/20 |
| Mean `J_eval` (intent-success) | 68.2 | 59.8 | 10.4 | 5.8 | **2.6** |
| Mean actuators | 9.0 | 9.0 | 9.0 | 3.2 | **2.4** |

Sparse voltage control is cheap. LLM-RA keeps `Σ|ΔP|≈0` on intent-success cases and uses fewer generators than OPF.

---

## Quick start

From the **repository root** (all paths are relative):

```bash
git clone https://github.com/JasonSDC/LLM_N1_Grid_Control.git
cd LLM_N1_Grid_Control

pip install -r requirements.txt

export ANTHROPIC_API_KEY='sk-ant-...'
python -m poc.run_poc

python -m unittest tests.test_retry_semantics
```

Needs Python 3.8+, `anthropic`, and a working `pandapower` OPF stack.

---

## Layout

```
.
├── README.md              this page
├── docs/paper.md          full Chinese write-up
├── poc/                   runnable package
│   ├── scenario.py        N-1 case generator
│   ├── physics_gateway.py violations, SLP, checks, fallback
│   ├── intent_engine.py   LLM + rule intent
│   ├── pipeline.py        main loop
│   ├── math_solver.py     OPF / SLSQP baselines
│   ├── metrics.py         J_eval
│   └── run_poc.py         experiment entry
├── tests/
└── requirements.txt
```

---

## Chinese notes

Method details, equations, and the same tables: [`docs/paper.md`](docs/paper.md).
