# GuardianZ: Real-Time Elder Scam & Financial Coercion Interceptor

IBM Z Datathon project. Standard fraud systems catch stolen cards; GuardianZ catches **authorised users being
manipulated into sending their savings**, as it happens.

```
Simulated stream ─► Scoring (behaviour + mule-graph + context) ─► Tier: allow / nudge / hold / escalate
                                   │                                        │
                            SHAP-style reasons              LLM cooling-off message + analyst brief
                                                                  + guardian alert
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py        # dashboard
pytest                      # sanity tests
```

The scaffold runs end-to-end today on **synthetic data with a rule-based stand-in model**. Optional local LLM:

```bash
ollama pull llama3.2:3b
GUARDIANZ_LLM=1 streamlit run app.py   # or use the sidebar toggle; falls back to templates on any failure
```

## Training the XGBoost model on PaySim

```bash
kaggle datasets download -d ealaxi/paysim1 -p data --unzip   # needs Kaggle API credentials
python -m core.train data/PS_20174392719_1491204439457_log.csv
git add models/ && git commit -m "Add trained behaviour model"
```

Big file / low RAM? Add `--neg-frac 0.05` to keep all fraud rows but only 5% of normal rows. To shrink the CSV for
sharing: `python -m core.paysim big.csv data/paysim_sample.csv 0.02` (a few MB; `git add -f` it).

Writes `models/behavior_xgb.json` and `models/behavior_metrics.json` (time-based split, PR-AUC / precision / recall).
`BehaviorModel` loads it automatically; without it the pipeline uses rule-based scoring. The model uses the signals
PaySim and a live event share (`amount_log`, `balance_pct`, `drains_account`, `high_risk_type`); reasons come from
exact TreeSHAP. After training, re-run `pytest` and the dashboard and re-tune tier thresholds if scams no longer reach hold.

### First PaySim training result (full 6.36M rows, time split)

Saved as `models/candidates/paysim_4feat_*` (NOT loaded by default; copy to `models/behavior_xgb.json` to try it).
PR-AUC **0.104** (test base rate 0.0034, so ~30x random), ROC-AUC 0.949; at p>=0.5 precision 13% / recall 24%.

- PaySim legit CASH_OUT/TRANSFER rows often empty the account too (1.19M rows; the "drains account" rule catches 98%
  of fraud but is only 0.7% precise), so these four live-available features are inherently weak.
- Fraud rate drifts over time (0.1% early, 1.5% in the last quarter), so time-split numbers are pessimistic.
- PaySim's balance-consistency columns (`oldbalanceOrg - amount - newbalanceOrig`, and the destination equivalent) give
  PR-AUC 1.0. That is a simulator artifact, not real-world signal, and not available at decision time: **don't use it**.

### Decision: XGBoost as one of three inputs

`models/behavior_xgb.json` (copy of the candidate) is now loaded by default. Fusion weights were set once, by judgement,
before evaluating: behaviour 0.20 / network 0.35 / context 0.45 (`core/scoring.py`). Measured result:

- Normal traffic (3,000 events): 0 false holds, 66 nudges (2.2%); p95 scoring latency ~1.3 ms.
- Impersonation, investment, tech-support: first transfer reaches **HOLD**. Later transfers in the same script often
  drop to NUDGE (in a real flow the first hold would stop the rest).
- **Romance: only NUDGE (known gap).** Its opening transfer has a new payee but no device change and a modest amount.
  Say so in the pitch rather than tuning around it.

### Repeated-payee signal

`PayeeHistory` (`core/features.py`) counts every transfer attempt per customer and payee (held ones too). If a customer
sends to a payee again within 30 minutes of first paying it, context score +25, and +15 more when the amount is at least
1.5x the previous one. Values were fixed before measuring. Measured effect:

- Normal traffic (3,000 events): still 0 false holds; nudges 66 -> 77 (2.2% -> 2.6%); ~1% of normal events carry the reason.
- Later transfers in scam scripts rise from ~40 to ~52 risk; tech-support's second transfer now reaches HOLD (80).
- **Romance still tops out at NUDGE** (risk 51-52, hold threshold 60). Its first transfer is the ambiguous one, and
  the signal only helps later transfers. Left as a documented gap rather than tuned.

## Who owns what

| Person | Files | First TODOs |
|---|---|---|
| **A: Data & ML** | `core/features.py`, `models.py`, `scoring.py` | Run `core.train` on PaySim and check metrics; add IsolationForest as an unsupervised signal; tune thresholds |
| **B: Streaming & graph** | `core/simulator.py`, `scams.py`, `graph.py` | `load_paysim`; richer scam scripts; learned/community mule signal; connect PaySim dest accounts to the graph |
| **C: GenAI & alerts** | `core/llm.py`, `guardian.py` | Prompt tuning; make LLM calls non-blocking (thread); real guardian channel (Telegram/email) |
| **D: UI & demo** | `app.py`, `ui/`, `core/pipeline.py` | Polish, LinuxONE numbers, demo script, slides |

**Contracts** (`core/schemas.py`): `Event` -> `ScoreResult` (+ `Intervention`, `GuardianAlert`). Change them only with
the whole team.

## Checkpoints

- **H12:** real data flowing through the real model into the UI.
- **H18:** feature freeze (core + scam simulator + mule graph + guardian alert). Stretch goals only after this.
- **H21:** demo rehearsal and backup video. **H23:** no new features.

## Honest notes for the pitch

- PaySim has no device / login / beneficiary fields; those are generated from a seed, and scam sequences are scripted.
- The mule graph is synthetic with planted rings.
- Latency numbers should be re-measured on the LinuxONE instance before presenting.
