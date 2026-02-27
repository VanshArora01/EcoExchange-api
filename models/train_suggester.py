
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def train_suggester():
    np.random.seed(42)
    data_path = os.path.join('data', 'suggestions_training.csv')
    save_dir = os.path.join('models', 'saved')
    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(data_path)
    
    # Feature engineering
    target_cols = [
        'suggest_night_shift', 'suggest_solar_ppa', 'suggest_power_factor_correction',
        'suggest_led_lighting', 'suggest_waste_heat_recovery', 'suggest_biogas_plant',
        'suggest_water_recycling', 'suggest_switch_to_png', 'suggest_renewable_procurement',
        'suggest_process_optimization'
    ]
    
    feature_cols = [
        'monthly_electricity_kwh', 'peak_demand_kw', 'power_factor', 
        'renewable_energy_percent', 'diesel_liters_monthly', 'natural_gas_kg_monthly',
        'coal_tons_monthly', 'organic_waste_kg_monthly', 'water_recycling_percent', 
        'water_consumption_kl_monthly', 'operating_hours_per_day', 'facility_area_sqm',
        'employee_total', 'scope1_kg', 'scope2_kg', 'emission_intensity',
        'industry', 'company_size'
    ]

    X = df[feature_cols].copy()
    y = df[target_cols].astype(np.int32)

    # Encoders
    le_industry = LabelEncoder()
    X['industry'] = le_industry.fit_transform(X['industry'])
    
    le_size = LabelEncoder()
    X['company_size'] = le_size.fit_transform(X['company_size'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model
    model = MultiOutputClassifier(
        XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracies = []
    for i, col in enumerate(target_cols):
        acc = accuracy_score(y_test.iloc[:, i], y_pred[:, i])
        accuracies.append(acc)
        print(f"Accuracy for {col}: {acc*100:.2f}%")

    avg_acc = np.mean(accuracies)
    
    # Save everything
    joblib.dump(model, os.path.join(save_dir, 'suggester_model.pkl'))
    joblib.dump(le_industry, os.path.join(save_dir, 'industry_encoder.pkl'))
    joblib.dump(le_size, os.path.join(save_dir, 'size_encoder.pkl'))
    joblib.dump(feature_cols, os.path.join(save_dir, 'feature_columns.pkl'))

    print(f"Suggester trained | Overall Accuracy: {avg_acc*100:.2f}% | Saved")

if __name__ == "__main__":
    try:
        train_suggester()
    except Exception as e:
        print(f"Error training suggester: {e}")
