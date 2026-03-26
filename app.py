import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────────────────────────────────────
# ALL 19 ARID & SEMI-ARID DISTRICTS — CAZRI / ICAR CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────
DISTRICTS_CONFIG = {
    # District            : (base_rainfall, base_gw_depth, base_temp, base_pop, base_agri)
    'Jaisalmer'          : (110,  38, 46.5,   600_000,  150_000),
    'Barmer'             : (140,  35, 46.0, 2_600_000,  400_000),
    'Bikaner'            : (135,  33, 45.5, 2_400_000,  420_000),
    'Jodhpur'            : (190,  28, 44.0, 3_700_000,  480_000),
    'Nagaur'             : (195,  26, 44.5, 3_300_000,  550_000),
    'Sri Ganganagar'     : (210,  22, 43.0, 1_900_000,  700_000),
    'Hanumangarh'        : (200,  20, 42.5, 1_800_000,  680_000),
    'Churu'              : (215,  24, 44.0, 2_000_000,  490_000),
    'Sikar'              : (220,  22, 43.5, 2_700_000,  500_000),
    'Jhunjhunu'          : (230,  20, 43.0, 2_200_000,  450_000),
    'Pali'               : (250,  25, 43.0, 2_000_000,  380_000),
    'Jalor'              : (260,  23, 43.0, 1_800_000,  360_000),
    'Ajmer'              : (300,  18, 41.5, 2_600_000,  420_000),
    'Bhilwara'           : (340,  18, 40.5, 2_500_000,  460_000),
    'Sirohi'             : (320,  20, 40.0, 1_100_000,  300_000),
    'Rajsamand'          : (360,  16, 39.5, 1_200_000,  280_000),
    'Tonk'               : (360,  15, 40.5, 1_500_000,  310_000),
    'Jaipur'             : (390,  14, 40.0, 6_700_000,  530_000),
    'Alwar'              : (590,  12, 38.5, 3_700_000,  490_000),
}

DISTRICTS = list(DISTRICTS_CONFIG.keys())

# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH   = 'model.pkl'
FEATURE_COLS = []
METRICS      = {}
HISTORY_JSON = {}

# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────
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
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILD
# ─────────────────────────────────────────────────────────────────────────────
def build_model(df):
    global FEATURE_COLS, METRICS
    df_enc = pd.get_dummies(df, columns=['District'], drop_first=True)
    X = df_enc.drop(columns=['Groundwater_Level_m'])
    y = df_enc['Groundwater_Level_m']
    FEATURE_COLS = list(X.columns)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    METRICS = {
        'rmse': round(float(np.sqrt(mean_squared_error(y_te, preds))), 3),
        'mae':  round(float(mean_absolute_error(y_te, preds)), 3),
        'r2':   round(float(r2_score(y_te, preds)), 4),
    }
    joblib.dump({'model': model, 'features': FEATURE_COLS}, MODEL_PATH)
    return model

def load_or_build():
    global FEATURE_COLS, METRICS, HISTORY_JSON
    csv_path = 'rajasthan_groundwater_data.csv'
    df = pd.read_csv(csv_path) if os.path.exists(csv_path) else create_synthetic_data()

    if os.path.exists(MODEL_PATH):
        pkg = joblib.load(MODEL_PATH)
        model        = pkg['model']
        FEATURE_COLS = pkg['features']
        df_enc = pd.get_dummies(df, columns=['District'], drop_first=True)
        X = df_enc[FEATURE_COLS]
        y = df_enc['Groundwater_Level_m']
        _, X_te, _, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        preds = model.predict(X_te)
        METRICS = {
            'rmse': round(float(np.sqrt(mean_squared_error(y_te, preds))), 3),
            'mae':  round(float(mean_absolute_error(y_te, preds)), 3),
            'r2':   round(float(r2_score(y_te, preds)), 4),
        }
    else:
        model = build_model(df)

    # History cache
    for dist in DISTRICTS:
        sub = df[df['District'] == dist].sort_values('Year')
        HISTORY_JSON[dist] = {
            'years':  sub['Year'].tolist(),
            'levels': sub['Groundwater_Level_m'].tolist(),
        }
    return model

MODEL = load_or_build()

# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/metrics')
def api_metrics():
    return jsonify(METRICS)

@app.route('/api/districts')
def api_districts():
    return jsonify(DISTRICTS)

@app.route('/api/history')
def api_history():
    district = request.args.get('district', DISTRICTS[0])
    return jsonify(HISTORY_JSON.get(district, {}))

@app.route('/api/all_history')
def api_all_history():
    """Returns 2025 snapshot for all districts — used for the comparison chart."""
    df = pd.read_csv('rajasthan_groundwater_data.csv')
    last = df[df['Year'] == 2025].groupby('District')['Groundwater_Level_m'].mean()
    result = [{'district': d, 'level': round(last.get(d, 0), 2)} for d in DISTRICTS]
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    body = request.get_json(force=True)
    try:
        district        = body['district']
        year            = int(body['year'])
        rainfall        = float(body['rainfall'])
        temperature     = float(body['temperature'])
        population      = float(body['population'])
        agriculture_area = float(body['agriculture_area'])
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400

    row = {
        'Year':              year,
        'Rainfall_mm':       rainfall,
        'Max_Temperature_C': temperature,
        'Population':        population,
        'Agriculture_Area_Ha': agriculture_area,
    }
    district_cols = [c for c in FEATURE_COLS if c.startswith('District_')]
    for col in district_cols:
        row[col] = 1 if district == col.replace('District_', '') else 0

    X_pred     = pd.DataFrame([row])[FEATURE_COLS]
    prediction = float(MODEL.predict(X_pred)[0])

    if prediction < 20:
        risk = 'Low'
    elif prediction < 35:
        risk = 'Moderate'
    elif prediction < 55:
        risk = 'High'
    else:
        risk = 'Critical'

    return jsonify({
        'groundwater_level_m': round(prediction, 2),
        'risk': risk,
        'district': district,
        'year': year,
    })

@app.route('/api/forecast')
def api_forecast():
    district = request.args.get('district', DISTRICTS[0])
    n_years  = min(int(request.args.get('years', 10)), 30)

    df   = pd.read_csv('rajasthan_groundwater_data.csv')
    last = df[df['District'] == district].sort_values('Year').iloc[-1]

    years, levels = [], []
    pop  = float(last['Population'])
    agri = float(last['Agriculture_Area_Ha'])
    rf   = float(last['Rainfall_mm'])
    tmp  = float(last['Max_Temperature_C'])

    for i in range(1, n_years + 1):
        year  = int(last['Year']) + i
        pop  *= 1.018
        agri *= 1.010
        rf    = max(10, rf - 1.2 + np.random.normal(0, 6))
        tmp  += 0.05

        row = {
            'Year': year, 'Rainfall_mm': rf,
            'Max_Temperature_C': tmp,
            'Population': pop,
            'Agriculture_Area_Ha': agri,
        }
        district_cols = [c for c in FEATURE_COLS if c.startswith('District_')]
        for col in district_cols:
            row[col] = 1 if district == col.replace('District_', '') else 0

        X_pred = pd.DataFrame([row])[FEATURE_COLS]
        pred   = float(MODEL.predict(X_pred)[0])
        years.append(year)
        levels.append(round(pred, 2))

    return jsonify({'years': years, 'levels': levels, 'district': district})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
