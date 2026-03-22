import joblib
import pandas as pd
import numpy as np
import warnings
import os

# Ignore the annoying feature name warnings
warnings.filterwarnings("ignore")

def run_test():
    print("\n--- 🛡️ Project Aegis: ML Logic Validation ---")
    
    # 1. Load the Model and the Scaler
    # Using relative paths to avoid the 'FileNotFound' ghost
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, 'models', 'fraud_model.pkl')
    scaler_path = os.path.join(base_dir, 'models', 'scaler.pkl')

    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        print("✅ Model and Scaler loaded successfully.\n")
    except Exception as e:
        print(f"❌ ERROR: Could not load model/scaler. Did you run train_model.py first?\n{e}")
        return

    # 2. Define Test Scenarios
    # Order must match your training features: [amount, hour, amount_diff_from_avg]
    test_scenarios = [
        {
            "name": "High-Value Late Night (High Risk)",
            "data": [120000, 3, 115000], # $120k at 3AM, $115k above user average
            "expected": "Fraud"
        },
        {
            "name": "Small Afternoon Coffee (Low Risk)",
            "data": [15, 14, 2],        # $15 at 2PM, $2 above user average
            "expected": "Safe"
        },
        {
            "name": "Medium Unusual Spike (Moderate)",
            "data": [5000, 11, 4800],    # $5k at 11AM, $4.8k above average
            "expected": "Review"
        }
    ]

    # 3. Execute Tests
    for test in test_scenarios:
        # Prepare the data as a DataFrame
        df = pd.DataFrame([test['data']], columns=['amount', 'hour', 'amount_diff_from_avg'])
        
        # IMPORTANT: We MUST scale the data using the saved scaler
        scaled_data = scaler.transform(df)
        
        # Get the Probability Score
        risk_score = model.predict_proba(scaled_data)[0][1] # Probability of being class 1 (Fraud)
        
        # Interpretation
        if risk_score > 0.7:
            status = "🚨 FRAUD DETECTED"
        elif risk_score > 0.4:
            status = "⚠️ SUSPICIOUS"
        else:
            status = "🟢 SAFE"
            
        print(f"Test: {test['name']}")
        print(f"   - Probability: {risk_score*100:.2f}%")
        print(f"   - Result: {status}")
        print("-" * 40)

if __name__ == "__main__":
    run_test()