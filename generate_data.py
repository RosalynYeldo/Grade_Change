import numpy as np
import pandas as pd

def generate_papermaking_data(num_transitions=50, timesteps_per_trans=120):
    np.random.seed(42)
    data = []
    
    for t_id in range(num_transitions):
        # Base grade parameters
        initial_bw = np.random.choice([60, 70, 80])
        target_bw = initial_bw + np.random.choice([-15, -10, 10, 15])
        
        speed = 800 + (100 - initial_bw) * 5
        stock_flow = initial_bw * 12.5
        filler_flow = stock_flow * 0.15
        steam_press = 2.5 + (initial_bw / 40)
        headbox_press = 1.2 * (speed / 800)**2
        retention_aid = 2.0 + np.random.normal(0, 0.1)

        for t in range(timesteps_per_trans):
            progress = t / timesteps_per_trans
            
            # Simulated ramp transition
            curr_target_bw = initial_bw + (target_bw - initial_bw) * (1 / (1 + np.exp(-10 * (progress - 0.5))))
            
            # Introduce non-linear transport delay & cross-loop disturbances
            curr_stock = stock_flow + (curr_target_bw * 12.5 - stock_flow) * progress
            curr_speed = speed - (curr_target_bw - initial_bw) * 3 * progress
            curr_filler = filler_flow + (curr_target_bw * 1.8 - filler_flow) * (progress ** 1.2)
            curr_steam = steam_press + (curr_target_bw / 40 - steam_press / 40) * progress
            
            # Process disturbances
            unmodeled_retention_dip = -0.4 if (0.3 < progress < 0.7 and t_id % 3 == 0) else 0.0
            curr_retention = retention_aid + unmodeled_retention_dip + np.random.normal(0, 0.02)
            
            # Basis Weight response formula (Physics + Disturbance)
            actual_bw = (curr_stock * 0.078 / (curr_speed / 800)) + (curr_retention * 0.8) + np.random.normal(0, 0.3)
            
            deviation_pct = abs(actual_bw - curr_target_bw) / curr_target_bw * 100
            at_risk = 1 if deviation_pct > 2.5 else 0
            
            data.append({
                "transition_id": t_id,
                "time_step": t,
                "stock_flow": curr_stock,
                "filler_flow": curr_filler,
                "machine_speed": curr_speed,
                "steam_pressure": curr_steam,
                "retention_aid_flow": curr_retention,
                "headbox_pressure": headbox_press,
                "target_basis_weight": curr_target_bw,
                "actual_basis_weight": actual_bw,
                "bw_deviation_pct": deviation_pct,
                "at_risk_2_5pct": at_risk
            })
            
    df = pd.DataFrame(data)
    df.to_csv("papermaking_historian_data.csv", index=False)
    print("Historian data generated successfully.")

if __name__ == "__main__":
    generate_papermaking_data()