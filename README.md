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

## Who owns what

| Person | Files | First TODOs |
|---|---|---|
| **A: Data & ML** | `core/features.py`, `models.py`, `scoring.py` | Train IsolationForest + XGBoost on PaySim; real SHAP in `BehaviorModel.predict`; tune thresholds; report PR-AUC |
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
