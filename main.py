import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────────────
# ALL ARID & SEMI-ARID DISTRICTS OF RAJASTHAN
# Source: CAZRI / ICAR classification
# ─────────────────────────────────────────────────────
DISTRICTS_CONFIG = {
    # District            : (base_rainfall, base_gw_depth, base_temp, base_pop, base_agri)
    # Hyper-arid / Core Thar
    'Jaisalmer'          : (110,  38, 46.5, 600_000,  150_000),
    'Barmer'             : (140,  35, 46.0, 2_600_000, 400_000),
    # Arid western
    'Bikaner'            : (135,  33, 45.5, 2_400_000, 420_000),
    'Jodhpur'            : (190,  28, 44.0, 3_700_000, 480_000),
    'Nagaur'             : (195,  26, 44.5, 3_300_000, 550_000),
    'Sri Ganganagar'     : (210,  22, 43.0, 1_900_000, 700_000),
    'Hanumangarh'        : (200,  20, 42.5, 1_800_000, 680_000),
    'Churu'              : (215,  24, 44.0, 2_000_000, 490_000),
    # Arid / semi-arid transition
    'Sikar'              : (220,  22, 43.5, 2_700_000, 500_000),
    'Jhunjhunu'          : (230,  20, 43.0, 2_200_000, 450_000),
    'Pali'               : (250,  25, 43.0, 2_000_000, 380_000),
    'Jalor'              : (260,  23, 43.0, 1_800_000, 360_000),
    # Semi-arid (CAZRI reclassified)
    'Ajmer'              : (300,  18, 41.5, 2_600_000, 420_000),
    'Bhilwara'           : (340,  18, 40.5, 2_500_000, 460_000),
    'Sirohi'             : (320,  20, 40.0, 1_100_000, 300_000),
    'Rajsamand'          : (360,  16, 39.5, 1_200_000, 280_000),
    'Tonk'               : (360,  15, 40.5, 1_500_000, 310_000),
    # Transitional semi-arid
    'Jaipur'             : (390,  14, 40.0, 6_700_000, 530_000),
    'Alwar'              : (590,  12, 38.5, 3_700_000, 490_000),
}

DISTRICTS = list(DISTRICTS_CONFIG.keys())


def create_synthetic_data():
    np.random.seed(42)
    rows = []

    for district, (base_rf, base_gw, base_tmp, base_pop, base_agri) in DISTRICTS_CONFIG.items():
        for year in range(1990, 2026):
            y = year - 1990
            rainfall   = max(10, base_rf + np.random.normal(0, 25) - y * 1.2)
            temp       = base_tmp + np.random.normal(0, 0.8) + y * 0.05
            population = base_pop  * (1.018 ** y)
            agri_area  = base_agri * (1.010 ** y)
            gw_level   = (base_gw
                          + (population / 500_000) * 1.0
                          + (agri_area  / 100_000) * 0.7
                          - (rainfall   / 100)     * 1.8
                          + y * 0.42
                          + np.random.normal(0, 0.4))
            rows.append({
                'District':            district,
                'Year':                year,
                'Rainfall_mm':         round(rainfall, 2),
                'Max_Temperature_C':   round(temp, 2),
                'Population':          int(population),
                'Agriculture_Area_Ha': int(agri_area),
                'Groundwater_Level_m': round(gw_level, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv('rajasthan_groundwater_data.csv', index=False)
    print(f"Dataset: {len(df)} rows | {df['District'].nunique()} districts")
    return df


def train_model_and_plot(df):
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # ── Historical trend (grouped)
    fig, axes = plt.subplots(4, 5, figsize=(22, 16), sharex=True)
    axes = axes.flatten()
    for i, dist in enumerate(DISTRICTS):
        sub = df[df['District'] == dist].sort_values('Year')
        axes[i].plot(sub['Year'], sub['Groundwater_Level_m'], color='steelblue', linewidth=1.8)
        axes[i].set_title(dist, fontsize=9)
        axes[i].grid(alpha=0.3)
    for j in range(len(DISTRICTS), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle('Groundwater Depletion Trends — All Arid Regions of Rajasthan (1990–2025)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('plots/all_districts_trend.png', dpi=120)
    plt.close()

    # ── 2025 depth comparison bar chart
    last_year = df[df['Year'] == 2025].groupby('District')['Groundwater_Level_m'].mean().sort_values(ascending=False)
    plt.figure(figsize=(14, 7))
    bars = plt.bar(last_year.index, last_year.values,
                   color=[f'#{int(255*(v/last_year.max())):02x}{int(80*(1-v/last_year.max())):02x}50'
                          for v in last_year.values])
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.ylabel('Groundwater Depth (m below surface)')
    plt.title('2025 Groundwater Depth Comparison — All 19 Arid Districts of Rajasthan')
    plt.tight_layout()
    plt.savefig('plots/district_comparison_2025.png', dpi=120)
    plt.close()

    # ── ML model
    df_enc = pd.get_dummies(df, columns=['District'], drop_first=True)
    X = df_enc.drop(columns=['Groundwater_Level_m'])
    y = df_enc['Groundwater_Level_m']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    mae  = mean_absolute_error(y_te, preds)

    # ── Actual vs Predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_te, preds, alpha=0.5, color='royalblue', s=12)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual (m)'); plt.ylabel('Predicted (m)')
    plt.title('AI Model: Predicted vs Actual Groundwater Depth')
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted.png', dpi=120)
    plt.close()

    # ── Feature importance
    importances = model.feature_importances_
    feats = X.columns
    idx = np.argsort(importances)[-15:]   # top 15
    plt.figure(figsize=(10, 7))
    plt.barh(range(len(idx)), importances[idx], align='center')
    plt.yticks(range(len(idx)), [feats[i] for i in idx])
    plt.title('Top Feature Importances for Groundwater Depletion')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=120)
    plt.close()

    return rmse, r2, mae


if __name__ == "__main__":
    print("Generating dataset for all 19 arid/semi-arid districts of Rajasthan...")
    df = create_synthetic_data()

    print("Training AI model and generating plots...")
    rmse, r2, mae = train_model_and_plot(df)

    print("\n===== Model Performance =====")
    print(f"  RMSE : {rmse:.4f} m")
    print(f"  MAE  : {mae:.4f} m")
    print(f"  R²   : {r2:.4f}")
    print("=============================")
    print("\nCharts saved to /plots. Run `python app.py` to launch the dashboard.")
