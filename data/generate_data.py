
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

def generate_data():
    np.random.seed(42)
    
    # Paths
    emissions_output = os.path.join('data', 'emissions_training.csv')
    suggestions_output = os.path.join('data', 'suggestions_training.csv')
    waste_output = os.path.join('data', 'waste_listings.csv')
    benchmarks_output = os.path.join('models', 'saved', 'benchmarks.json')
    
    os.makedirs('data', exist_ok=True)
    os.makedirs(os.path.join('models', 'saved'), exist_ok=True)

    industries = [
        'Steel', 'Energy', 'Chemical', 'Automotive', 'Manufacturing',
        'Textile', 'Food_Processing', 'Pharmaceutical', 'Electronics', 'Construction'
    ]
    
    industry_ranges = {
        'Steel': (80000, 200000),
        'Energy': (150000, 400000),
        'Chemical': (60000, 150000),
        'Automotive': (40000, 120000),
        'Manufacturing': (30000, 90000),
        'Textile': (150000, 50000), # Wait, the user said Textile: 15,000 - 50,000. Typo in my prompt check.
        'Food_Processing': (10000, 40000),
        'Pharmaceutical': (8000, 30000),
        'Electronics': (5000, 20000),
        'Construction': (20000, 60000)
    }
    # Fixing Textile range from user prompt: 15,000 - 50,000
    industry_ranges['Textile'] = (15000, 50000)

    size_multipliers = {
        'Startup': 0.1, 'SME': 0.3, 'Mid': 0.6, 'Large': 1.0, 'Enterprise': 2.5
    }
    company_sizes = list(size_multipliers.keys())

    # Seasonal multipliers
    def get_seasonal_multiplier(month):
        if month in [4, 5, 6]: return 1.15  # Summer
        if month in [7, 8, 9]: return 0.95  # Monsoon
        if month in [11, 12, 1]: return 1.05 # Winter
        return 1.0 # Spring/Other

    # 1. Generate emissions_training.csv (8000 rows: 400 companies * 20 months)
    emissions_data = []
    num_companies = 400
    months_per_company = 20
    
    for comp_idx in range(num_companies):
        company_id = f"C{comp_idx+1:03d}"
        industry = industries[comp_idx // 40]
        size = np.random.choice(company_sizes)
        
        # Base CO2 for month 1
        low, high = industry_ranges[industry]
        base_co2 = np.random.uniform(low, high) * size_multipliers[size]
        
        current_co2 = base_co2
        
        for m_idx in range(1, months_per_company + 1):
            # Month-over-month continuity
            if m_idx > 1:
                current_co2 = current_co2 * (1 + np.random.uniform(-0.05, 0.08))
            
            # Seasonal effect
            month_val = (m_idx - 1) % 12 + 1
            seasonal_mult = get_seasonal_multiplier(month_val)
            total_co2_kg = current_co2 * seasonal_mult
            
            # Back-calculate consistent inputs
            # Assuming total_co2 = scope1 + scope2 + scope3
            # scope1 = diesel + nat_gas + coal + others (simplified)
            # scope2 = electricity
            # scope3 = raw_mat + prod_vol
            
            s1_ratio = np.random.uniform(0.3, 0.5)
            s2_ratio = np.random.uniform(0.3, 0.4)
            s3_ratio = 1.0 - s1_ratio - s2_ratio
            
            scope1_kg = total_co2_kg * s1_ratio
            scope2_kg = total_co2_kg * s2_ratio
            scope3_kg = total_co2_kg * s3_ratio
            
            # Breakdown Scope 1
            diesel = (scope1_kg * 0.4) / 2.68
            nat_gas = (scope1_kg * 0.3) / 2.75
            coal = (scope1_kg * 0.3) / (1000 * 2.86)
            
            # Breakdown Scope 2
            renewable_pct = np.random.uniform(0, 30)
            electricity_kwh = scope2_kg / (0.82 * (1 - renewable_pct/100))
            
            # Breakdown Scope 3
            prod_vol = scope3_kg * 0.1
            raw_mat = scope3_kg * 0.005
            
            # Other fields
            employees = int(np.random.uniform(50, 5000) * size_multipliers[size])
            if employees < 1: employees = 1
            
            emissions_data.append({
                'company_id': company_id,
                'month_index': m_idx,
                'industry': industry,
                'company_size': size,
                'employee_total': employees,
                'facility_area_sqm': round(employees * np.random.uniform(10, 50), 1),
                'operating_hours_per_day': np.random.randint(8, 25),
                'operating_days_per_week': np.random.randint(5, 8),
                'number_of_shifts': np.random.randint(1, 4),
                'monthly_electricity_kwh': round(electricity_kwh, 2),
                'peak_demand_kw': round(electricity_kwh * 0.015, 2),
                'power_factor': round(np.random.uniform(0.85, 0.98), 3),
                'renewable_energy_percent': round(renewable_pct, 2),
                'electricity_provider': np.random.choice(['Tata Power', 'Adani Electricity', 'MSEDCL', 'BESCOM']),
                'diesel_liters_monthly': round(diesel, 2),
                'petrol_liters_monthly': round(np.random.uniform(50, 200), 2),
                'natural_gas_kg_monthly': round(nat_gas, 2),
                'lpg_kg_monthly': round(np.random.uniform(100, 500), 2),
                'coal_tons_monthly': round(coal, 4),
                'furnace_oil_liters_monthly': round(np.random.uniform(0, 500), 2),
                'biomass_tons_monthly': 0,
                'water_consumption_kl_monthly': round(employees * 0.5, 2),
                'water_recycling_percent': round(np.random.uniform(0, 50), 2),
                'metal_scrap_kg_monthly': round(total_co2_kg * 0.05, 1),
                'plastic_waste_kg_monthly': round(total_co2_kg * 0.02, 1),
                'organic_waste_kg_monthly': round(total_co2_kg * 0.01, 1),
                'paper_waste_kg_monthly': round(total_co2_kg * 0.01, 1),
                'wood_waste_kg_monthly': 0,
                'production_volume_monthly': round(prod_vol, 1),
                'raw_material_consumption_tons': round(raw_mat, 2),
                'avg_temperature': 25 + np.random.uniform(-5, 10),
                'scope1_kg': round(scope1_kg, 2),
                'scope2_kg': round(scope2_kg, 2),
                'scope3_kg': round(scope3_kg, 2),
                'total_co2_kg': round(total_co2_kg, 2),
                'emission_intensity': round(total_co2_kg / employees, 2)
            })

    df_emissions = pd.DataFrame(emissions_data)
    df_emissions.to_csv(emissions_output, index=False)
    print(f"emissions_training.csv created: {len(df_emissions)} rows")

    # 2. Generate suggestions_training.csv (5000 rows)
    # Using a similar continuity and consistency approach
    suggestions_data = []
    for i in range(5000):
        # Sample an existing row or create a representative one
        base_row = df_emissions.iloc[np.random.randint(0, len(df_emissions))].to_dict()
        
        # Targets based on logic
        row = base_row.copy()
        row['suggest_night_shift'] = 1 if row['operating_hours_per_day'] > 18 and row['monthly_electricity_kwh'] > 50000 else 0
        row['suggest_solar_ppa'] = 1 if row['renewable_energy_percent'] < 10 else 0
        row['suggest_power_factor_correction'] = 1 if row['power_factor'] < 0.92 else 0
        row['suggest_led_lighting'] = 1 if row['facility_area_sqm'] > 5000 and row['monthly_electricity_kwh'] > 20000 else 0
        row['suggest_waste_heat_recovery'] = 1 if row['industry'] in ['Steel', 'Energy', 'Chemical'] and row['coal_tons_monthly'] > 2 else 0
        row['suggest_biogas_plant'] = 1 if row['organic_waste_kg_monthly'] > 200 else 0
        row['suggest_water_recycling'] = 1 if row['water_recycling_percent'] < 20 and row['water_consumption_kl_monthly'] > 100 else 0
        row['suggest_switch_to_png'] = 1 if row['diesel_liters_monthly'] > 500 else 0
        row['suggest_renewable_procurement'] = 1 if row['scope2_kg'] > 10000 and row['renewable_energy_percent'] < 15 else 0
        row['suggest_process_optimization'] = 1 if row['emission_intensity'] > 50 else 0
        
        # Add some noise to make ML work
        for col in [c for c in row.keys() if c.startswith('suggest_')]:
            if np.random.rand() < 0.05:
                row[col] = 1 - row[col]
                
        suggestions_data.append(row)
        
    df_suggestions = pd.DataFrame(suggestions_data)
    df_suggestions.to_csv(suggestions_output, index=False)
    print(f"suggestions_training.csv created: {len(df_suggestions)} rows")

    # 3. Generate waste_listings.csv (3000 rows)
    material_dist = {
        'Metal_Scrap': 0.30,
        'Plastic': 0.15,
        'Organic': 0.15,
        'Paper': 0.10,
        'Construction_Debris': 0.10,
        'E_Waste': 0.05,
        'Wood': 0.05,
        'Glass': 0.05,
        'Textile': 0.05
    }
    
    price_ranges = {
        'Metal_Scrap': (15, 40),
        'Plastic': (5, 20),
        'Paper': (2, 8),
        'Organic': (0, 3),
        'E_Waste': (20, 80),
        'Construction_Debris': (1, 5),
        'Wood': (2, 6),
        'Glass': (3, 10),
        'Textile': (4, 12)
    }

    clusters = [
        {"name": "Industrial belt", "lat": (20, 25), "lon": (75, 82), "weight": 0.40},
        {"name": "North India", "lat": (27, 32), "lon": (74, 80), "weight": 0.25},
        {"name": "South India", "lat": (10, 18), "lon": (76, 82), "weight": 0.20},
        {"name": "East India", "lat": (20, 27), "lon": (82, 90), "weight": 0.15}
    ]

    waste_data = []
    materials = list(material_dist.keys())
    material_probs = list(material_dist.values())
    
    for i in range(3000):
        mat_type = np.random.choice(materials, p=material_probs)
        cluster = np.random.choice(clusters, p=[c['weight'] for c in clusters])
        
        lat = np.random.uniform(cluster['lat'][0], cluster['lat'][1])
        lon = np.random.uniform(cluster['lon'][0], cluster['lon'][1])
        
        pr_low, pr_high = price_ranges.get(mat_type, (5, 15))
        
        waste_data.append({
            'listing_id': f"L{i+1:04d}",
            'seller_id': f"C{np.random.randint(1, 500):03d}",
            'material_type': mat_type,
            'quantity_kg': np.random.randint(50, 5000),
            'price_per_kg': round(np.random.uniform(pr_low, pr_high), 2),
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'industry': np.random.choice(industries),
            'quality_grade': np.random.choice(['A', 'B', 'C']),
            'available_monthly': np.random.rand() < 0.75,
            'created_at': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat()
        })
        
    df_waste = pd.DataFrame(waste_data)
    df_waste.to_csv(waste_output, index=False)
    print(f"waste_listings.csv created: {len(df_waste)} rows")

    # 4. Generate benchmarks.json (Enhanced for new endpoint)
    benchmarks = {}
    for industry in industries:
        id_df = df_emissions[df_emissions['industry'] == industry]
        benchmarks[industry] = {
            'industry': industry,
            'avg_monthly_co2_kg': round(float(id_df['total_co2_kg'].mean()), 2),
            'best_in_class_co2_kg': round(float(id_df['total_co2_kg'].min()), 2),
            'worst_in_class_co2_kg': round(float(id_df['total_co2_kg'].max()), 2),
            'avg_emission_intensity': round(float(id_df['emission_intensity'].mean()), 2),
            'top_emission_sources': [
                "electricity", "diesel_combustion", "natural_gas"
            ] if industry not in ['Steel', 'Energy'] else ["coal_combustion", "electricity", "furnace_oil"],
            'common_optimizations': [
                "renewable_procurement", "led_lighting"
            ] if industry not in ['Steel', 'Chemical'] else ["waste_heat_recovery", "process_optimization"],
            'companies_in_dataset': 40,
            'data_last_updated': "2024-01-01"
        }
    
    with open(benchmarks_output, 'w') as f:
        json.dump(benchmarks, f, indent=4)
    print(f"benchmarks.json created for {len(industries)} industries")

if __name__ == "__main__":
    generate_data()
