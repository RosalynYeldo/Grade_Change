# 🧻 GradeIntel AI: Grade Change Intelligence System

An intelligent automatic grade change system for paper making processes that predicts process deviations ($>2.5\%$ on Basis Weight) occurring during grade transitions and provides real-time, explainable setpoint recommendations.

---

##  Problem Overview
During grade transitions in paper mills, operators manage dynamic interactions across stock flow, filler flow, steam pressure, machine speed, and moisture. Traditional Quality Control Systems (QCS) execute coordinated ramps but do not learn from historical transition data, leading to off-spec paper, broke material, and extended stabilization times.

**GradeIntel AI** sits above the Honeywell QCS layer to:
1. **Predict Risk:** Identify when Basis Weight is at risk of deviating $>2.5\%$ from setpoint before quality limits are exceeded.
2. **Recommend Setpoints:** Offer optimal setpoints to maintain safe operating limits and accelerate stabilization.
3. **Uncover Hidden Correlations:** Identify unmodeled parameter couplings (e.g., retention aid flow vs. basis weight variance).
4. **Explainable AI (XAI):** Provide SHAP-based root cause attributions for operator transparency.
5. **Human-in-the-Loop (HITL):** Record operator accept/reject actions to fine-tune future model calibrations.

---

##  System Architecture
```text
[ QCS / Historian Trends ] ──► [ Data Ingestion & Lag Alignment ]
│
▼
[ Operator Feedback Log ]  ◄── [ Streamlit Dashboard ] ◄── [ XGBoost Risk Model & SHAP Explainer ]
│
▼
[ Setpoint Optimizer ]

```
##  Getting Started

### 1. Prerequisites & Installation
Ensure you have Python 3.9+ installed. Install project dependencies:

pip install -r requirements.txt

2. Generate Historical Process Data
Simulate paper making transition data from historian and QCS logs:

Bash
python generate_data.py
3. Launch the Dashboard
Run the Streamlit application to launch the interactive real-time dashboard:

streamlit run app.py

## Dashboard Key Features
Real-time Trajectory Forecast: Visualizes Basis Weight against upper/lower 2.5% quality thresholds.

SHAP Root Cause Attribution: Ranks process variables driving the predicted deviation risk.

Cross-Correlation Discovery: Highlights non-obvious loop interactions affecting system stability.

Interactive Setpoint Nudges: Allows operators to accept or reject AI recommendations with feedback logging.

📁 Repository Structure
├── app.py                 # Streamlit Real-Time Dashboard
├── generate_data.py       # Paper Making Historian Simulator
├── model_engine.py        # ML Engine, SHAP Analysis & Setpoint Optimizer
├── requirements.txt       # Python Dependencies
└── README.md              # Project Documentation

---

