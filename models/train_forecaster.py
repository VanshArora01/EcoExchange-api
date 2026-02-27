
import pandas as pd
import numpy as np
import os
import json
import glob

def train_forecasters():
    # Paths
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_path = os.path.join(root_dir, 'data', 'emissions_training.csv')
    save_dir = os.path.join(root_dir, 'models', 'saved')
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Error: Training data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Ensure numeric columns
    cols = ['total_co2_kg', 'scope1_kg', 'scope2_kg', 'scope3_kg']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    industries = df['industry'].unique().tolist()
    industry_stats = {}

    for industry in industries:
        ind_df = df[df['industry'] == industry].copy()
        
        # 1. avg_monthly_co2
        avg_monthly_co2 = float(ind_df['total_co2_kg'].mean())
        
        # 2. std_monthly_co2
        std_monthly_co2 = float(ind_df['total_co2_kg'].std())
        
        # 3. seasonal_multipliers (dict of month 1-12 to multiplier)
        # Extract calendar month (1-12) from month_index
        ind_df['month_num'] = ((ind_df['month_index'] - 1) % 12) + 1
        
        monthly_avg = ind_df.groupby('month_num')['total_co2_kg'].mean()
        overall_avg = ind_df['total_co2_kg'].mean()
        
        seasonal_multipliers = {}
        for m in range(1, 13):
            if m in monthly_avg.index:
                seasonal_multipliers[str(m)] = float(round(monthly_avg[m] / overall_avg, 3))
            else:
                seasonal_multipliers[str(m)] = 1.0
        
        # 4. growth_rate (average month-over-month change)
        company_growth_rates = []
        for cid in ind_df['company_id'].unique():
            comp_df = ind_df[ind_df['company_id'] == cid].sort_values('month_index')
            if len(comp_df) > 1:
                # Calculate MoM change: (curr - prev) / prev
                pct_change = comp_df['total_co2_kg'].pct_change().dropna()
                # Remove outliers/inf if any
                pct_change = pct_change[np.isfinite(pct_change)]
                if not pct_change.empty:
                    company_growth_rates.append(pct_change.mean())
        
        growth_rate = float(np.mean(company_growth_rates)) if company_growth_rates else 0.0
        
        # 5, 6, 7. scope ratios
        # We handle cases where total might be zero to avoid div by zero
        temp_df = ind_df[ind_df['total_co2_kg'] > 0].copy()
        if not temp_df.empty:
            scope1_ratio = float((temp_df['scope1_kg'] / temp_df['total_co2_kg']).mean())
            scope2_ratio = float((temp_df['scope2_kg'] / temp_df['total_co2_kg']).mean())
            scope3_ratio = float((temp_df['scope3_kg'] / temp_df['total_co2_kg']).mean())
        else:
            scope1_ratio = 0.4
            scope2_ratio = 0.4
            scope3_ratio = 0.2

        # 8, 9. min/max (5th and 95th percentiles)
        min_co2 = float(np.percentile(ind_df['total_co2_kg'], 5))
        max_co2 = float(np.percentile(ind_df['total_co2_kg'], 95))

        industry_stats[industry] = {
            "avg_monthly_co2": round(avg_monthly_co2, 2),
            "std_monthly_co2": round(std_monthly_co2, 2),
            "seasonal_multipliers": seasonal_multipliers,
            "growth_rate": round(growth_rate, 4),
            "scope1_ratio": round(scope1_ratio, 4),
            "scope2_ratio": round(scope2_ratio, 4),
            "scope3_ratio": round(scope3_ratio, 4),
            "min_co2": round(min_co2, 2),
            "max_co2": round(max_co2, 2)
        }
        
        print(f"{industry} | avg: {industry_stats[industry]['avg_monthly_co2']} | std: {industry_stats[industry]['std_monthly_co2']} | growth: {industry_stats[industry]['growth_rate']*100:.2f}% | Saved")

    # Save to JSON
    json_path = os.path.join(save_dir, 'industry_stats.json')
    with open(json_path, 'w') as f:
        json.dump(industry_stats, f, indent=2)
    
    print("All industry stats saved to industry_stats.json")

    # Delete old Prophet .pkl files
    pkl_files = glob.glob(os.path.join(save_dir, "prophet_*.pkl"))
    for f in pkl_files:
        try:
            os.remove(f)
            # print(f"Deleted {f}")
        except:
            pass
    
    # Also delete industry_list.pkl if it exists as it was used by the old Prophet logic
    list_pkl = os.path.join(save_dir, "industry_list.pkl")
    if os.path.exists(list_pkl):
        os.remove(list_pkl)

if __name__ == "__main__":
    train_forecasters()
