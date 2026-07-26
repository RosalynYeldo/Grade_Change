import pandas as pd
import numpy as np
import xgboost as xgb
import shap

class GradeChangeIntelligenceEngine:
    def __init__(self, data_path="papermaking_historian_data.csv"):
        self.df = pd.read_csv(data_path)
        self.features = [
            "stock_flow", "filler_flow", "machine_speed", 
            "steam_pressure", "retention_aid_flow", "headbox_pressure"
        ]
        self.target = "at_risk_2_5pct"
        self.model = None
        self.explainer = None
        
    def train_predictive_model(self):
        X = self.df[self.features]
        y = self.df[self.target]
        
        self.model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
        )
        self.model.fit(X, y)
        self.explainer = shap.TreeExplainer(self.model)
        
    def find_discovered_correlations(self):
        """Finds unexpected or unmodeled correlations between secondary loops and Basis Weight deviation."""
        corr_matrix = self.df[self.features + ["bw_deviation_pct"]].corr()
        bw_corrs = corr_matrix["bw_deviation_pct"].drop("bw_deviation_pct").sort_values(key=abs, ascending=False)
        return bw_corrs

    def predict_and_explain(self, current_state_dict):
        input_df = pd.DataFrame([current_state_dict])[self.features]
        risk_prob = self.model.predict_proba(input_df)[0][1]
        shap_vals = self.explainer.shap_values(input_df)[0]
        
        contributions = dict(zip(self.features, shap_vals))
        top_driver = max(contributions, key=lambda k: abs(contributions[k]))
        
        return risk_prob, contributions, top_driver

    def recommend_corrective_setpoints(self, current_state):
        """Calculates setpoint nudges to bring predicted Basis Weight within +/- 2.5% specification."""
        rec = current_state.copy()
        
        # Calculate ideal stock flow based on speed and target basis weight
        target_bw = current_state["target_basis_weight"]
        ideal_stock = (target_bw * (current_state["machine_speed"] / 800)) / 0.078
        
        rec["stock_flow"] = round(ideal_stock, 2)
        
        # Adjust retention aid if retention flow is causing instability
        if current_state["retention_aid_flow"] < 1.9:
            rec["retention_aid_flow"] = 2.15
            
        return rec