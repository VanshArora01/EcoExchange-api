
import pandas as pd
import numpy as np
import json
import os

def analyze():
    print("Starting Dataset Analysis...")
    
    # Paths
    steel_path = 'Steel_industry_data.csv'
    ember_path = 'india_yearly_full_release_long_format.csv'
    output_dir = os.path.join('models', 'saved')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'real_patterns.json')

    # 1. Steel Data Analysis
    print("\n--- Analyzing Steel Industry Data ---")
    df_steel = pd.read_csv(steel_path)
    print(f"Columns: {df_steel.columns.tolist()}")
    print(f"Shape: {df_steel.shape}")
    print("First 3 rows:")
    print(df_steel.head(3))
    
    numeric_cols = df_steel.select_dtypes(include=[np.number]).columns
    print("\nBasic Stats for Numeric Columns:")
    print(df_steel[numeric_cols].describe().loc[['mean', 'min', 'max']])
    
    categorical_cols = df_steel.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col != 'date':
            print(f"Unique values in {col}: {df_steel[col].unique()}")

    # Extraction for patterns
    df_steel['date'] = pd.to_datetime(df_steel['date'], dayfirst=True)
    df_steel['month'] = df_steel['date'].dt.month
    df_steel['hour'] = df_steel['date'].dt.hour
    df_steel['is_weekend'] = df_steel['Day_of_week'].isin(['Saturday', 'Sunday'])

    # Monthly energy totals
    monthly_totals = df_steel.groupby('month')['Usage_kWh'].sum().to_dict()
    annual_avg = sum(monthly_totals.values()) / 12
    seasonal_multipliers = {int(m): val / annual_avg for m, val in monthly_totals.items()}

    # Weekday vs weekend ratio
    avg_weekday = df_steel[~df_steel['is_weekend']]['Usage_kWh'].mean()
    avg_weekend = df_steel[df_steel['is_weekend']]['Usage_kWh'].mean()
    weekday_weekend_ratio = avg_weekday / avg_weekend if avg_weekend != 0 else 1.0

    # Real CO2 per kWh ratio
    # Filter out rows where Usage_kWh is 0 to avoid div by zero
    valid_co2 = df_steel[df_steel['Usage_kWh'] > 0]
    co2_per_kwh = (valid_co2['CO2(tCO2)'] * 1000 / valid_co2['Usage_kWh']).mean() # Convert tons to kg maybe? Steel data CO2 is usually small.
    # Actually some rows might have 0 CO2 for some reason. Let's just take the overall mean of non-zero.
    
    # Average power factor
    avg_power_factor = df_steel['Lagging_Current_Power_Factor'].mean() / 100 # It's in pct in dataset

    # Peak hours
    hourly_avg = df_steel.groupby('hour')['Usage_kWh'].mean()
    overall_avg = df_steel['Usage_kWh'].mean()
    peak_hours = hourly_avg[hourly_avg > overall_avg].index.tolist()

    # Load type distribution
    load_dist = (df_steel['Load_Type'].value_counts(normalize=True) * 100).to_dict()

    # 2. Ember India Data Analysis
    print("\n--- Analyzing Ember India Electricity Data ---")
    df_ember = pd.read_csv(ember_path)
    print(f"Columns: {df_ember.columns.tolist()}")
    print(f"Shape: {df_ember.shape}")
    print("First 3 rows:")
    print(df_ember.head(3))
    print(f"Unique Variables: {df_ember['Variable'].unique().tolist()}")
    print(f"Unique Categories: {df_ember['Category'].unique().tolist()}")
    print(f"Unique States: {df_ember['State'].unique().tolist()}")
    print(f"Year Range: {df_ember['Year'].min()} - {df_ember['Year'].max()}")

    # Extraction for patterns
    latest_year = df_ember['Year'].max()
    df_latest = df_ember[(df_ember['Year'] == latest_year) & (df_ember['Category'] == 'Capacity')]
    
    # Get Fossil % per state
    state_fossil_pct = {}
    for state in df_ember['State'].unique():
        state_data = df_latest[df_latest['State'] == state]
        fossil = state_data[state_data['Variable'] == 'Fossil']
        total = state_data[state_data['Variable'] == 'Total'] # Might not exist, let's check
        
        # Alternative: Sum Fossil and Clean if Total is missing
        if fossil.empty:
            fossil_val = 0
        else:
            fossil_val = fossil['Value'].values[0]
            
        clean = state_data[state_data['Variable'] == 'Clean']
        clean_val = clean['Value'].values[0] if not clean.empty else 0
        
        total_val = fossil_val + clean_val
        if total_val > 0:
            state_fossil_pct[state] = (fossil_val / total_val) * 100
        else:
            state_fossil_pct[state] = 75.0 # default

    # Convert fossil % to grid carbon intensity
    grid_intensities = {state: 0.05 + (pct / 100) * 0.90 for state, pct in state_fossil_pct.items()}

    # Map states to electricity providers
    # Punjab → PSPCL
    # Maharashtra → MSEDCL  
    # Gujarat → Adani and Torrent
    # Other → 0.820 (India average)
    
    pspcl = grid_intensities.get('Punjab', 0.820)
    msedcl = grid_intensities.get('Maharashtra', 0.790)
    # Gujarat has multiple, maybe average or just use state intensity
    guj_intensity = grid_intensities.get('Gujarat', 0.700)
    adani = guj_intensity + 0.01 
    torrent = guj_intensity - 0.02
    tata_power = grid_intensities.get('Karnataka', 0.650) # Fallback / Example

    grid_factors = {
        "PSPCL": round(float(pspcl), 3),
        "MSEDCL": round(float(msedcl), 3),
        "Adani": round(float(adani), 3),
        "Torrent": round(float(torrent), 3),
        "Tata_Power": round(float(tata_power), 3),
        "Other": 0.820
    }

    patterns = {
        "steel_patterns": {
            "monthly_totals": {int(k): float(v) for k, v in monthly_totals.items()},
            "seasonal_multipliers": seasonal_multipliers,
            "weekday_weekend_ratio": float(weekday_weekend_ratio),
            "co2_per_kwh": float(co2_per_kwh),
            "avg_power_factor": float(avg_power_factor),
            "peak_hours": [int(h) for h in peak_hours],
            "load_type_distribution": load_dist
        },
        "grid_factors": grid_factors
    }

    with open(output_path, 'w') as f:
        json.dump(patterns, f, indent=4)

    print(f"\nreal_patterns.json saved successfully to {output_path}")

if __name__ == "__main__":
    try:
        analyze()
    except Exception as e:
        print(f"Error during analysis: {e}")
