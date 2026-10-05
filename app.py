import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
import json
import io
from typing import Optional, Tuple, Dict, Any, List
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Optional imports with fallback
try:
    from streamlit_echarts import st_echarts
    ECHARTS_AVAILABLE = True
except ImportError:
    ECHARTS_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EnvHealth IQ Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0f1e;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00d4aa;
    --accent2: #6366f1;
    --warn: #f59e0b;
    --danger: #ef4444;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e293b;
}

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0f172a 50%, #0a0f1e 100%) !important;
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #111827 100%) !important;
    border-right: 1px solid #1e2d45 !important;
}

/* Header */
.dash-header {
    background: linear-gradient(90deg, rgba(0,212,170,0.08) 0%, rgba(99,102,241,0.08) 100%);
    border: 1px solid rgba(0,212,170,0.2);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4aa, #6366f1, #00d4aa);
}
.dash-title {
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #00d4aa, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 !important;
}
.dash-subtitle {
    color: #64748b;
    font-size: 0.9rem;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

/* Metric cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: linear-gradient(135deg, #111827, #1a2235);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    opacity: 0;
    transition: opacity 0.3s;
}
.metric-card:hover::after { opacity: 1; }
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #00d4aa;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Risk badges */
.risk-high   { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.4);  color: #ef4444; padding: 3px 10px; border-radius: 20px; font-size:0.8rem; font-weight:600; }
.risk-mod    { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); color: #f59e0b; padding: 3px 10px; border-radius: 20px; font-size:0.8rem; font-weight:600; }
.risk-low    { background: rgba(0,212,170,0.15);  border: 1px solid rgba(0,212,170,0.4);  color: #00d4aa; padding: 3px 10px; border-radius: 20px; font-size:0.8rem; font-weight:600; }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    color: #00d4aa;
    border-left: 3px solid #00d4aa;
    padding-left: 12px;
    margin: 24px 0 16px 0;
    letter-spacing: 0.3px;
}

/* Alert boxes */
.alert-high { background: rgba(239,68,68,0.08); border-left: 4px solid #ef4444; border-radius: 8px; padding: 16px; margin: 8px 0; }
.alert-mod  { background: rgba(245,158,11,0.08); border-left: 4px solid #f59e0b; border-radius: 8px; padding: 16px; margin: 8px 0; }
.alert-low  { background: rgba(0,212,170,0.08);  border-left: 4px solid #00d4aa; border-radius: 8px; padding: 16px; margin: 8px 0; }
.alert-info { background: rgba(99,102,241,0.08); border-left: 4px solid #6366f1; border-radius: 8px; padding: 16px; margin: 8px 0; }

/* Data table */
.stDataFrame { border-radius: 8px; overflow: hidden; }
[data-testid="stDataFrame"] { border: 1px solid #1e293b !important; border-radius: 8px !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,212,170,0.15), rgba(99,102,241,0.15)) !important;
    border: 1px solid rgba(0,212,170,0.3) !important;
    color: #00d4aa !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,212,170,0.25), rgba(99,102,241,0.25)) !important;
    border-color: #00d4aa !important;
    transform: translateY(-1px);
}

/* Selectbox, text inputs */
.stSelectbox > div > div, .stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* Progress bars */
.stProgress > div > div > div { background: linear-gradient(90deg, #00d4aa, #6366f1) !important; border-radius: 4px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background: #111827 !important; border-radius: 10px !important; border: 1px solid #1e293b !important; }
.stTabs [data-baseweb="tab"] { color: #64748b !important; font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; }
.stTabs [aria-selected="true"] { color: #00d4aa !important; }
.stTabs [data-baseweb="tab-highlight"] { background: #00d4aa !important; }

/* Expander */
.streamlit-expanderHeader { background: #111827 !important; border: 1px solid #1e293b !important; border-radius: 8px !important; color: #e2e8f0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0f1e; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #00d4aa; }

/* Footer */
.footer {
    text-align: center;
    color: #334155;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    padding: 24px 0 8px 0;
    border-top: 1px solid #1e293b;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_api_key():
    try:
        return st.secrets["OPENWEATHER_API_KEY"]
    except:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if api_key:
            return api_key
        return "9b99c2520d9ddb36ed867de4196e0ede"

api = get_api_key()

# ─────────────────────────────────────────────────────────────
# FALLBACK COORDINATES
# ─────────────────────────────────────────────────────────────
fallback_coords = {
    "pathankot":  (32.2643, 75.6421),
    "jammu":      (32.7266, 74.8570),
    "shimla":     (31.1048, 77.1734),
    "gurdaspur":  (32.0409, 75.4061),
    "gurugram":   (28.4595, 77.0266),
    "delhi":      (28.6139, 77.2090),
    "mumbai":     (19.0760, 72.8777),
    "bangalore":  (12.9716, 77.5946),
    "chennai":    (13.0827, 80.2707),
    "kolkata":    (22.5726, 88.3639),
    "hyderabad":  (17.3850, 78.4867),
    "pune":       (18.5204, 73.8567),
    "ahmedabad":  (23.0225, 72.5714),
    "jaipur":     (26.9124, 75.7873),
    "lucknow":    (26.8467, 80.9462),
    "chandigarh": (30.7333, 76.7794),
    "amritsar":   (31.6340, 74.8723),
    "ludhiana":   (30.9010, 75.8573),
}

# ─────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_coordinates(city: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={city},IN&limit=1&appid={api}"
        response = requests.get(geo_url, timeout=10)
        response.raise_for_status()
        geo_data = response.json()
        if geo_data:
            return geo_data[0]["lat"], geo_data[0]["lon"]
        elif city.lower() in fallback_coords:
            return fallback_coords[city.lower()]
        return None, None
    except:
        if city.lower() in fallback_coords:
            return fallback_coords[city.lower()]
        return None, None

def calculate_aqi(concentration, breakpoints):
    if concentration is None or concentration < 0:
        return None
    for bp_low, bp_high, aqi_low, aqi_high in breakpoints:
        if bp_low <= concentration <= bp_high:
            return round(((aqi_high - aqi_low) / (bp_high - bp_low)) * (concentration - bp_low) + aqi_low)
    return 500

def calculate_pm25_aqi(c):
    if c is None: return None
    bp = [(0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
          (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]
    return calculate_aqi(c, bp)

def calculate_pm10_aqi(c):
    if c is None: return None
    bp = [(0,54,0,50),(55,154,51,100),(155,254,101,150),
          (255,354,151,200),(355,424,201,300),(425,504,301,400),(505,604,401,500)]
    return calculate_aqi(c, bp)

def calculate_o3_aqi(c):
    if c is None: return None
    ppb = c / 1.96
    bp = [(0,54,0,50),(55,70,51,100),(71,85,101,150),
          (86,105,151,200),(106,200,201,300),(201,404,301,400),(405,604,401,500)]
    return calculate_aqi(ppb, bp)

def calculate_no2_aqi(c):
    if c is None: return None
    ppb = c / 1.88
    bp = [(0,53,0,50),(54,100,51,100),(101,360,101,150),
          (361,649,151,200),(650,1249,201,300),(1250,1649,301,400),(1650,2049,401,500)]
    return calculate_aqi(ppb, bp)

def calculate_so2_aqi(c):
    if c is None: return None
    ppb = c / 2.62
    bp = [(0,35,0,50),(36,75,51,100),(76,185,101,150),
          (186,304,151,200),(305,604,201,300),(605,804,301,400),(805,1004,401,500)]
    return calculate_aqi(ppb, bp)

def calculate_co_aqi(c):
    if c is None: return None
    ppm = c / 1145
    bp = [(0.0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),
          (12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,40.4,301,400),(40.5,50.4,401,500)]
    return calculate_aqi(ppm, bp)

def calculate_comprehensive_aqi(components):
    aqi_values = {}
    mappings = {'pm2_5': calculate_pm25_aqi, 'pm10': calculate_pm10_aqi,
                'o3': calculate_o3_aqi, 'no2': calculate_no2_aqi,
                'so2': calculate_so2_aqi, 'co': calculate_co_aqi}
    label_map = {'pm2_5': 'PM2.5', 'pm10': 'PM10', 'o3': 'O3', 'no2': 'NO2', 'so2': 'SO2', 'co': 'CO'}
    for key, fn in mappings.items():
        if key in components and components[key] is not None:
            val = fn(components[key])
            if val is not None:
                aqi_values[label_map[key]] = val
    if not aqi_values:
        return {'overall_aqi': None, 'dominant_pollutant': None, 'individual_aqis': {}}
    overall = max(aqi_values.values())
    dominant = max(aqi_values, key=aqi_values.get)
    return {'overall_aqi': overall, 'dominant_pollutant': dominant, 'individual_aqis': aqi_values}

@st.cache_data(ttl=300, show_spinner=False)
def get_comprehensive_aqi_data(lat, lon, api_key):
    try:
        url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={api_key}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "list" not in data or not data["list"]:
            return {'error': 'No AQI data'}
        components = data["list"][0]["components"]
        result = calculate_comprehensive_aqi(components)
        return {
            'components': components,
            'aqi_result': result,
            'api_aqi': data["list"][0].get("main", {}).get("aqi", "N/A"),
            'timestamp': data["list"][0].get("dt", None)
        }
    except Exception as e:
        return {'error': str(e)}

def get_aqi_category(aqi):
    if aqi is None: return "Unknown", "#64748b"
    elif aqi <= 50:  return "Good", "#00d4aa"
    elif aqi <= 100: return "Moderate", "#f59e0b"
    elif aqi <= 150: return "Unhealthy for Sensitive", "#f97316"
    elif aqi <= 200: return "Unhealthy", "#ef4444"
    elif aqi <= 300: return "Very Unhealthy", "#9333ea"
    else:            return "Hazardous", "#7f1d1d"

@st.cache_data(ttl=600, show_spinner=False)
def get_weather_data(city, api_key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {'error': str(e)}

@st.cache_data(ttl=600, show_spinner=False)
def get_current_weather(city, api_key):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {'error': str(e)}

@st.cache_data(show_spinner=False)
def load_disease_data(file_path):
    try:
        df = pd.read_csv(file_path)
        required = ["City","Disease","Date","Population_Affected","Number_of_Deaths","Survived","Doctors_Available","Hospitals_Available"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return pd.DataFrame(), []
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Month"] = df["Date"].dt.month
        df["Season"] = df["Month"].apply(lambda m:
            "Winter" if m in [12,1,2] else
            "Summer" if m in [3,4,5] else
            "Monsoon" if m in [6,7,8,9] else "Post-Monsoon")
        df["CFR"] = (df["Number_of_Deaths"] / df["Population_Affected"].replace(0, np.nan) * 100).round(2)
        df["Survival_Rate"] = (df["Survived"] / df["Population_Affected"].replace(0, np.nan) * 100).round(2)
        return df, sorted(df["City"].unique())
    except FileNotFoundError:
        st.warning(f"CSV file '{file_path}' not found.")
        return pd.DataFrame(), []
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), []

def compute_disease_risk(temp, humidity, rain, wind_speed, latest_aqi, latest_pm25, latest_pm10, dominant_pollutant, temp_change):
    results = []

    # DENGUE
    s, f = 0, []
    if 25 <= temp <= 32: s += 3; f.append("Optimal mosquito temp")
    elif 20 <= temp <= 35: s += 2; f.append("Favorable temp")
    elif temp > 35: s += 1; f.append("High temp")
    if humidity > 80: s += 3; f.append("Very high humidity")
    elif humidity > 70: s += 2; f.append("High humidity")
    elif humidity > 60: s += 1; f.append("Moderate humidity")
    if rain > 15: s += 3; f.append("Heavy rainfall")
    elif rain > 5: s += 2; f.append("Moderate rainfall")
    elif rain > 1: s += 1; f.append("Light rainfall")
    if wind_speed < 1: s += 1; f.append("Low wind")
    if latest_aqi > 100: s += 1; f.append("Poor AQI")
    risk = "High" if s >= 6 else "Moderate" if s >= 3 else "Low"
    results.append({"Disease":"Dengue","Risk":risk,"Score":s,"Max_Score":11,"Factors":f})

    # HEAT STROKE
    s, f = 0, []
    if temp > 42: s += 4; f.append("Extreme heat")
    elif temp > 38: s += 3; f.append("Very high temp")
    elif temp > 35: s += 2; f.append("High temp")
    elif temp > 32: s += 1; f.append("Warm temp")
    if humidity > 80: s += 3; f.append("Very high humidity")
    elif humidity > 70: s += 2; f.append("High humidity")
    elif humidity > 50: s += 1; f.append("Moderate humidity")
    if wind_speed < 1: s += 2; f.append("No wind")
    elif wind_speed < 3: s += 1; f.append("Low wind")
    heat_index = temp + 0.5 * humidity / 100 * temp
    if heat_index > 40: s += 2; f.append(f"Heat index {heat_index:.1f}°C")
    risk = "High" if s >= 7 else "Moderate" if s >= 3 else "Low"
    results.append({"Disease":"Heat Stroke","Risk":risk,"Score":s,"Max_Score":12,"Factors":f})

    # ASTHMA
    s, f = 0, []
    if latest_aqi > 200: s += 4; f.append("Very unhealthy AQI")
    elif latest_aqi > 150: s += 3; f.append("Unhealthy AQI")
    elif latest_aqi > 100: s += 2; f.append("Sensitive AQI")
    elif latest_aqi > 50: s += 1; f.append("Moderate AQI")
    if latest_pm25 > 75: s += 3; f.append("Very high PM2.5")
    elif latest_pm25 > 55: s += 2; f.append("High PM2.5")
    elif latest_pm25 > 35: s += 1; f.append("Elevated PM2.5")
    if latest_pm10 > 150: s += 2; f.append("Very high PM10")
    elif latest_pm10 > 100: s += 1; f.append("High PM10")
    if dominant_pollutant in ["O3","NO2","SO2"]: s += 2; f.append(f"Dominant: {dominant_pollutant}")
    if temp_change > 15: s += 3; f.append("Very high temp variation")
    elif temp_change > 10: s += 2; f.append("High temp variation")
    elif temp_change > 5: s += 1; f.append("Moderate temp variation")
    risk = "High" if s >= 8 else "Moderate" if s >= 4 else "Low"
    results.append({"Disease":"Asthma","Risk":risk,"Score":s,"Max_Score":15,"Factors":f})

    # RESPIRATORY INFECTIONS
    s, f = 0, []
    if latest_aqi > 150: s += 3; f.append("Poor AQI")
    elif latest_aqi > 100: s += 2; f.append("Moderate AQI")
    if temp < 15: s += 2; f.append("Cold temp")
    elif temp > 35: s += 1; f.append("Hot temp stress")
    if humidity > 80: s += 2; f.append("Very high humidity")
    elif humidity < 30: s += 1; f.append("Very dry air")
    if temp_change > 10: s += 2; f.append("High temp variation")
    if wind_speed < 1: s += 1; f.append("Poor ventilation")
    risk = "High" if s >= 5 else "Moderate" if s >= 3 else "Low"
    results.append({"Disease":"Respiratory Infections","Risk":risk,"Score":s,"Max_Score":10,"Factors":f})

    # MALARIA
    s, f = 0, []
    if 20 <= temp <= 30: s += 3; f.append("Optimal malaria temp")
    elif 15 <= temp <= 35: s += 1; f.append("Favorable temp")
    if humidity > 60: s += 2; f.append("High humidity")
    if rain > 10: s += 3; f.append("Heavy rainfall")
    elif rain > 3: s += 2; f.append("Moderate rainfall")
    if wind_speed < 2: s += 1; f.append("Low wind")
    risk = "High" if s >= 6 else "Moderate" if s >= 3 else "Low"
    results.append({"Disease":"Malaria","Risk":risk,"Score":s,"Max_Score":9,"Factors":f})

    # CHOLERA / WATERBORNE
    s, f = 0, []
    if rain > 20: s += 3; f.append("Heavy flooding rain")
    elif rain > 10: s += 2; f.append("Heavy rain")
    if temp > 25: s += 2; f.append("Warm temp")
    if humidity > 70: s += 1; f.append("High humidity")
    if latest_aqi > 100: s += 1; f.append("Poor sanitation proxy")
    risk = "High" if s >= 5 else "Moderate" if s >= 3 else "Low"
    results.append({"Disease":"Waterborne Diseases","Risk":risk,"Score":s,"Max_Score":7,"Factors":f})

    for r in results:
        r["Risk_Pct"] = round(r["Score"] / r["Max_Score"] * 100, 1)

    return results

# ─────────────────────────────────────────────────────────────
# ML MODEL
# ─────────────────────────────────────────────────────────────
def train_ml_model(disease_df):
    if not ML_AVAILABLE or disease_df.empty:
        return None, None, None, None

    df = disease_df.copy()
    le_city = LabelEncoder()
    le_disease = LabelEncoder()
    le_season = LabelEncoder()

    df["City_enc"]    = le_city.fit_transform(df["City"])
    df["Disease_enc"] = le_disease.fit_transform(df["Disease"])
    df["Season_enc"]  = le_season.fit_transform(df["Season"])

    features = ["City_enc","Disease_enc","Season_enc","Month","Doctors_Available","Hospitals_Available"]
    available = [f for f in features if f in df.columns]

    df_clean = df[available + ["Population_Affected"]].dropna()
    if len(df_clean) < 10:
        return None, None, None, None

    X = df_clean[available]
    y = df_clean["Population_Affected"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    encoders = {"city": le_city, "disease": le_disease, "season": le_season}
    return model, encoders, available, {"mae": mae, "r2": r2, "test_size": len(X_test)}

def predict_cases(model, encoders, feature_names, city, disease, season, month, doctors, hospitals):
    if model is None: return None
    try:
        row = {}
        if "City_enc"    in feature_names: row["City_enc"]    = encoders["city"].transform([city])[0]
        if "Disease_enc" in feature_names: row["Disease_enc"] = encoders["disease"].transform([disease])[0]
        if "Season_enc"  in feature_names: row["Season_enc"]  = encoders["season"].transform([season])[0]
        if "Month"             in feature_names: row["Month"]             = month
        if "Doctors_Available" in feature_names: row["Doctors_Available"] = doctors
        if "Hospitals_Available" in feature_names: row["Hospitals_Available"] = hospitals
        X = pd.DataFrame([row])[feature_names]
        return int(model.predict(X)[0])
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────
# PDF REPORT GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_pdf_report(city, risk_results, env_data, disease_summary=None):
    if not PDF_AVAILABLE:
        return None
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=40, leftMargin=40,
                            topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('Title', parent=styles['Title'],
        fontSize=20, spaceAfter=6,
        textColor=colors.HexColor('#00d4aa'), alignment=TA_CENTER)
    sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
        fontSize=11, spaceAfter=4,
        textColor=colors.HexColor('#64748b'), alignment=TA_CENTER)
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
        fontSize=13, spaceBefore=16, spaceAfter=6,
        textColor=colors.HexColor('#6366f1'))
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
        fontSize=10, spaceAfter=4, leading=14)

    story.append(Paragraph("EnvHealth IQ — Disease Risk Report", title_style))
    story.append(Paragraph(f"City: {city}  |  Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1e293b')))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Environmental Conditions", h2_style))
    env_rows = [["Parameter", "Value"]] + [[k, v] for k, v in env_data.items()]
    env_table = Table(env_rows, colWidths=[200, 200])
    env_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a2235')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#00d4aa')),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#0d1526'), colors.HexColor('#111827')]),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#e2e8f0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#1e293b')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(env_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Disease Risk Assessment", h2_style))
    risk_rows = [["Disease", "Risk Level", "Score (%)", "Key Factors"]]
    for r in risk_results:
        factors = ", ".join(r.get("Factors", [])[:3])
        risk_rows.append([r["Disease"], r["Risk"], f"{r['Risk_Pct']}%", factors])
    risk_table = Table(risk_rows, colWidths=[120, 80, 70, 200])
    risk_color_map = {"High": colors.HexColor('#ef4444'), "Moderate": colors.HexColor('#f59e0b'), "Low": colors.HexColor('#00d4aa')}
    risk_style = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a2235')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.HexColor('#00d4aa')),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#0d1526'), colors.HexColor('#111827')]),
        ('TEXTCOLOR', (0,1), (-1,-1), colors.HexColor('#e2e8f0')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#1e293b')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]
    for i, r in enumerate(risk_results, 1):
        c = risk_color_map.get(r["Risk"], colors.gray)
        risk_style.append(('TEXTCOLOR', (1, i), (1, i), c))
        risk_style.append(('FONTNAME', (1, i), (1, i), 'Helvetica-Bold'))
    risk_table.setStyle(TableStyle(risk_style))
    story.append(risk_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Health Recommendations", h2_style))
    high_risks = [r for r in risk_results if r["Risk"] == "High"]
    if high_risks:
        story.append(Paragraph(f"⚠ HIGH ALERT: {', '.join([r['Disease'] for r in high_risks])}", 
            ParagraphStyle('Alert', parent=body_style, textColor=colors.HexColor('#ef4444'), fontName='Helvetica-Bold')))
        story.append(Paragraph("Immediate action required. Consult healthcare professionals and follow local health authority guidelines.", body_style))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Disclaimer: This report is based on environmental data and statistical models. Always consult qualified healthcare professionals for medical decisions.", 
        ParagraphStyle('Disc', parent=body_style, textColor=colors.HexColor('#64748b'), fontSize=8)))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ─────────────────────────────────────────────────────────────
# ECHARTS HELPER
# ─────────────────────────────────────────────────────────────
def render_chart(options, height="400px"):
    if ECHARTS_AVAILABLE:
        st_echarts(options=options, height=height)
    else:
        st.json(options)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
disease_df, disease_cities = load_disease_data("output_d206b0_corrected.csv")

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem; color: #00d4aa; font-weight:700;'>🌍 EnvHealth IQ</div>
        <div style='font-size: 0.7rem; color: #334155; margin-top: 2px;'>Environmental Disease Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**📍 Location Settings**")
    weather_aqi_city = st.text_input("Weather / AQI City", "Pathankot", key="weather_city")
    disease_city = st.selectbox("Disease Data City", disease_cities if disease_cities else ["No Data"], key="disease_city")

    st.markdown("---")
    st.markdown("**📋 Dashboard Sections**")
    show_home        = st.checkbox("🏠 Home Overview",       value=True)
    show_weather     = st.checkbox("🌦 Weather Data",        value=False)
    show_aqi         = st.checkbox("💨 AQI Analysis",        value=False)
    show_disease     = st.checkbox("🩺 Disease Statistics",  value=False)
    show_compare     = st.checkbox("🔁 City Comparison",     value=False)
    show_correlation = st.checkbox("📊 Correlation & Risk",  value=False)
    show_seasonal    = st.checkbox("🍂 Seasonal Analysis",   value=False)
    show_ml          = st.checkbox("🤖 ML Prediction",       value=False)
    show_map         = st.checkbox("🗺 India Risk Map",       value=False)
    show_report      = st.checkbox("📄 Download Report",     value=False)

    st.markdown("---")
    if st.button("🔄 Refresh All Data"):
        st.cache_data.clear()
        st.rerun()

    st.markdown("""
    <div style='font-size: 0.65rem; color: #334155; text-align:center; margin-top: 8px;'>
        Data via OpenWeatherMap API<br>
        Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-header'>
    <div class='dash-title'>🌍 EnvHealth IQ Dashboard</div>
    <div class='dash-subtitle'>Environment–Disease Intelligence Platform for Indian Cities</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────
if show_home:
    col1, col2, col3, col4 = st.columns(4)
    total_cities    = len(disease_cities) if disease_cities else 0
    total_diseases  = disease_df["Disease"].nunique() if not disease_df.empty else 0
    total_cases     = int(disease_df["Population_Affected"].sum()) if not disease_df.empty else 0
    total_deaths    = int(disease_df["Number_of_Deaths"].sum()) if not disease_df.empty else 0

    with col1:
        st.markdown(f"""<div class='metric-card'><div class='metric-value'>{total_cities}</div><div class='metric-label'>Cities Tracked</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'><div class='metric-value'>{total_diseases}</div><div class='metric-label'>Diseases Monitored</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'><div class='metric-value'>{total_cases:,}</div><div class='metric-label'>Total Cases</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'><div class='metric-value'>{total_deaths:,}</div><div class='metric-label'>Total Deaths</div></div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📋 Platform Overview</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='alert-info'>
        <strong style='color:#6366f1;'>🎯 What This Dashboard Does</strong><br><br>
        Correlates real-time environmental conditions with disease outbreak risks across Indian cities.
        Combines live weather data, air quality metrics, and historical disease records to generate
        actionable health intelligence.
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='alert-low'>
        <strong style='color:#00d4aa;'>🔬 How Risk Is Calculated</strong><br><br>
        Each disease has a weighted scoring system based on scientifically established environmental triggers.
        Scores are normalized to a percentage, then classified as Low / Moderate / High using
        validated epidemiological thresholds.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🧬 Target Diseases & Triggers</div>", unsafe_allow_html=True)
    disease_info = {
        "🦟 Dengue":              "Rainfall · Humidity > 70% · Temp 25-32°C · Stagnant water",
        "🌡️ Heat Stroke":         "Temp > 35°C · Humidity > 70% · Low wind · High heat index",
        "🫁 Asthma":              "PM2.5/PM10 · Ozone · NO₂ · Temperature swings",
        "🤧 Respiratory Infections": "Cold weather · High humidity · Poor AQI · Temp variation",
        "🦠 Malaria":             "Temp 20-30°C · Rainfall > 10mm · High humidity",
        "💧 Waterborne Diseases": "Heavy flooding rain · Warm temperatures · Poor sanitation",
    }
    cols = st.columns(3)
    for i, (disease, factors) in enumerate(disease_info.items()):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left; margin-bottom:8px;'>
                <div style='font-family: Space Mono, monospace; font-size:0.9rem; color:#e2e8f0; margin-bottom:6px;'>{disease}</div>
                <div style='font-size:0.75rem; color:#64748b; line-height:1.5;'>{factors}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🗺 How to Navigate</div>", unsafe_allow_html=True)
    steps = [
        ("1️⃣ Set Location", "Enter your city in the sidebar for live weather & AQI data"),
        ("2️⃣ Choose City Data", "Select a city from the disease dataset dropdown"),
        ("3️⃣ Enable Sections", "Toggle sections in the sidebar to explore different views"),
        ("4️⃣ Run ML Prediction", "Use the ML tab to predict future outbreak cases"),
        ("5️⃣ Download Report", "Generate a PDF health risk report for your city"),
    ]
    cols = st.columns(5)
    for i, (step, desc) in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left;'>
                <div style='color:#00d4aa; font-family: Space Mono, monospace; font-size:0.8rem; margin-bottom:4px;'>{step}</div>
                <div style='font-size:0.72rem; color:#64748b;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# WEATHER
# ─────────────────────────────────────────────────────────────
if show_weather:
    st.markdown("<div class='section-header'>🌦 Weather Forecast</div>", unsafe_allow_html=True)
    with st.spinner(f"Fetching weather for {weather_aqi_city}..."):
        wdata = get_weather_data(weather_aqi_city, api)
    if 'error' in wdata:
        st.error(f"Weather error: {wdata['error']}")
    elif "list" not in wdata:
        st.error("No weather data returned.")
    else:
        rows = []
        for entry in wdata["list"]:
            rows.append({
                "Date & Time": datetime.fromtimestamp(entry["dt"]).strftime("%Y-%m-%d %H:%M"),
                "Temp (°C)": round(entry["main"]["temp"], 1),
                "Feels Like (°C)": round(entry["main"]["feels_like"], 1),
                "Humidity (%)": entry["main"]["humidity"],
                "Wind (m/s)": entry.get("wind", {}).get("speed", 0),
                "Description": entry["weather"][0]["description"].title(),
                "Rain (mm)": entry.get("rain", {}).get("3h", 0),
            })
        wdf = pd.DataFrame(rows)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🌡 Current Temp", f"{wdf['Temp (°C)'].iloc[0]}°C")
        c2.metric("💧 Humidity", f"{wdf['Humidity (%)'].iloc[0]}%")
        c3.metric("💨 Wind Speed", f"{wdf['Wind (m/s)'].iloc[0]} m/s")
        c4.metric("🌧 Rain (3h)", f"{wdf['Rain (mm)'].iloc[0]} mm")

        with st.expander("📋 Full Forecast Table"):
            st.dataframe(wdf, use_container_width=True)

        tabs = st.tabs(["📈 Temp & Humidity", "🌧 Rainfall", "💨 Wind"])
        with tabs[0]:
            chart = {
                "tooltip": {"trigger": "axis"},
                "legend": {"data": ["Temperature", "Humidity"], "textStyle": {"color": "#e2e8f0"}},
                "backgroundColor": "transparent",
                "xAxis": {"type": "category", "data": wdf["Date & Time"].tolist(), "axisLabel": {"color": "#64748b", "rotate": 30}},
                "yAxis": {"type": "value", "axisLabel": {"color": "#64748b"}},
                "series": [
                    {"name": "Temperature", "type": "line", "smooth": True,
                     "data": wdf["Temp (°C)"].tolist(),
                     "lineStyle": {"color": "#ef4444"}, "itemStyle": {"color": "#ef4444"},
                     "areaStyle": {"color": "rgba(239,68,68,0.1)"}},
                    {"name": "Humidity", "type": "line", "smooth": True,
                     "data": wdf["Humidity (%)"].tolist(),
                     "lineStyle": {"color": "#6366f1"}, "itemStyle": {"color": "#6366f1"},
                     "areaStyle": {"color": "rgba(99,102,241,0.1)"}},
                ]
            }
            render_chart(chart)

        with tabs[1]:
            rain_chart = {
                "tooltip": {"trigger": "axis"},
                "backgroundColor": "transparent",
                "xAxis": {"type": "category", "data": wdf["Date & Time"].tolist(), "axisLabel": {"color": "#64748b", "rotate": 30}},
                "yAxis": {"type": "value", "name": "mm", "axisLabel": {"color": "#64748b"}},
                "series": [{"name": "Rain", "type": "bar", "data": wdf["Rain (mm)"].tolist(),
                            "itemStyle": {"color": "#00d4aa"}}]
            }
            render_chart(rain_chart)

        with tabs[2]:
            wind_chart = {
                "tooltip": {"trigger": "axis"},
                "backgroundColor": "transparent",
                "xAxis": {"type": "category", "data": wdf["Date & Time"].tolist(), "axisLabel": {"color": "#64748b", "rotate": 30}},
                "yAxis": {"type": "value", "name": "m/s", "axisLabel": {"color": "#64748b"}},
                "series": [{"name": "Wind Speed", "type": "line", "smooth": True,
                            "data": wdf["Wind (m/s)"].tolist(),
                            "lineStyle": {"color": "#f59e0b"}, "itemStyle": {"color": "#f59e0b"},
                            "areaStyle": {"color": "rgba(245,158,11,0.1)"}}]
            }
            render_chart(wind_chart)

# ─────────────────────────────────────────────────────────────
# AQI
# ─────────────────────────────────────────────────────────────
if show_aqi:
    st.markdown("<div class='section-header'>💨 Air Quality Index Analysis</div>", unsafe_allow_html=True)
    lat, lon = get_coordinates(weather_aqi_city)
    if lat is None:
        st.error(f"Could not get coordinates for {weather_aqi_city}")
    else:
        with st.spinner("Fetching AQI data..."):
            aqi_data = get_comprehensive_aqi_data(lat, lon, api)

        if 'error' in aqi_data:
            st.error(aqi_data['error'])
        else:
            components  = aqi_data['components']
            aqi_result  = aqi_data['aqi_result']
            calc_aqi    = aqi_result['overall_aqi']
            dominant    = aqi_result['dominant_pollutant'] or "N/A"
            category, color = get_aqi_category(calc_aqi)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🧮 Calculated AQI", calc_aqi or "N/A")
            c2.metric("🏆 Dominant Pollutant", dominant)
            c3.metric("📊 Category", category)
            c4.metric("🌐 API Scale", f"{aqi_data['api_aqi']}/5")

            st.markdown(f"""<div class='alert-{"high" if calc_aqi and calc_aqi > 150 else "mod" if calc_aqi and calc_aqi > 100 else "low"}'>
            <strong>AQI Status:</strong> {calc_aqi} — <strong>{category}</strong> &nbsp;
            Dominant pollutant: <strong>{dominant}</strong>
            </div>""", unsafe_allow_html=True)

            if aqi_result['individual_aqis']:
                pollutant_data = []
                for p, v in aqi_result['individual_aqis'].items():
                    cat, _ = get_aqi_category(v)
                    conc = components.get(p.lower().replace('.','_'), 0)
                    pollutant_data.append({"Pollutant": p, "AQI": v, "Category": cat, "Concentration (µg/m³)": f"{conc:.1f}"})
                st.dataframe(pd.DataFrame(pollutant_data), use_container_width=True)

                bar_data = []
                colors_list = []
                for p, v in aqi_result['individual_aqis'].items():
                    _, c = get_aqi_category(v)
                    bar_data.append({"value": v, "itemStyle": {"color": c}})

                bar_chart = {
                    "tooltip": {"trigger": "axis"},
                    "backgroundColor": "transparent",
                    "xAxis": {"type": "category", "data": list(aqi_result['individual_aqis'].keys()), "axisLabel": {"color": "#64748b"}},
                    "yAxis": {"type": "value", "name": "AQI", "axisLabel": {"color": "#64748b"}},
                    "series": [{"type": "bar", "data": bar_data,
                                "label": {"show": True, "position": "top", "color": "#e2e8f0"}}]
                }
                render_chart(bar_chart)

            if calc_aqi and calc_aqi > 0:
                gauge = {
                    "backgroundColor": "transparent",
                    "series": [{
                        "type": "gauge", "min": 0, "max": 500,
                        "detail": {"formatter": str(calc_aqi), "color": "#e2e8f0", "fontSize": 24},
                        "data": [{"value": calc_aqi, "name": "AQI"}],
                        "title": {"color": "#64748b"},
                        "axisLine": {"lineStyle": {"color": [
                            [0.1, "#00d4aa"],[0.2, "#f59e0b"],[0.3, "#f97316"],
                            [0.4, "#ef4444"],[0.6, "#9333ea"],[1, "#7f1d1d"]
                        ]}}
                    }]
                }
                render_chart(gauge, "350px")

# ─────────────────────────────────────────────────────────────
# DISEASE STATISTICS
# ─────────────────────────────────────────────────────────────
if show_disease:
    st.markdown("<div class='section-header'>🩺 Disease Statistics</div>", unsafe_allow_html=True)
    if disease_df.empty:
        st.error("Disease dataset not loaded. Please check the CSV file.")
    else:
        city_data = disease_df[disease_df["City"].str.lower() == disease_city.lower()]
        if city_data.empty:
            st.error(f"No data found for {disease_city}")
        else:
            diseases = city_data["Disease"].unique()

            # Summary metrics
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("🏙 City", disease_city)
            c2.metric("🦠 Diseases", len(diseases))
            c3.metric("👥 Total Cases", f"{int(city_data['Population_Affected'].sum()):,}")
            c4.metric("💀 Deaths", f"{int(city_data['Number_of_Deaths'].sum()):,}")
            avg_cfr = city_data['CFR'].mean()
            c5.metric("📉 Avg CFR", f"{avg_cfr:.1f}%" if not np.isnan(avg_cfr) else "N/A")

            for disease in diseases:
                ddata = city_data[city_data["Disease"] == disease]
                with st.expander(f"🧬 {disease} — {int(ddata['Population_Affected'].sum()):,} total cases", expanded=True):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Cases",    f"{int(ddata['Population_Affected'].sum()):,}")
                    c2.metric("Deaths",   f"{int(ddata['Number_of_Deaths'].sum()):,}")
                    c3.metric("Survived", f"{int(ddata['Survived'].sum()):,}")
                    avg_cfr_d = ddata['CFR'].mean()
                    c4.metric("CFR", f"{avg_cfr_d:.1f}%" if not np.isnan(avg_cfr_d) else "N/A")

                    if len(ddata) > 1:
                        ddata_sorted = ddata.sort_values("Date")
                        trend = {
                            "tooltip": {"trigger": "axis"},
                            "legend": {"data": ["Cases","Deaths","Survived"], "textStyle": {"color": "#e2e8f0"}},
                            "backgroundColor": "transparent",
                            "xAxis": {"type": "category", "data": ddata_sorted["Date"].astype(str).tolist(), "axisLabel": {"color": "#64748b", "rotate": 30}},
                            "yAxis": {"type": "value", "axisLabel": {"color": "#64748b"}},
                            "series": [
                                {"name": "Cases",    "type": "line", "smooth": True, "data": ddata_sorted["Population_Affected"].tolist(), "lineStyle": {"color": "#6366f1"}, "itemStyle": {"color": "#6366f1"}},
                                {"name": "Deaths",   "type": "line", "smooth": True, "data": ddata_sorted["Number_of_Deaths"].tolist(),   "lineStyle": {"color": "#ef4444"}, "itemStyle": {"color": "#ef4444"}},
                                {"name": "Survived", "type": "line", "smooth": True, "data": ddata_sorted["Survived"].tolist(),           "lineStyle": {"color": "#00d4aa"}, "itemStyle": {"color": "#00d4aa"}},
                            ]
                        }
                        render_chart(trend)

# ─────────────────────────────────────────────────────────────
# CITY COMPARISON
# ─────────────────────────────────────────────────────────────
if show_compare:
    st.markdown("<div class='section-header'>🔁 City Comparison</div>", unsafe_allow_html=True)
    if disease_df.empty:
        st.error("Disease data required for comparison.")
    else:
        available_cities = sorted(disease_df["City"].unique())
        selected = st.multiselect("Select 2-4 cities to compare", available_cities,
                                  default=available_cities[:min(3, len(available_cities))])
        if len(selected) < 2:
            st.info("Please select at least 2 cities.")
        else:
            comp_data = disease_df[disease_df["City"].isin(selected)]
            city_summary = comp_data.groupby("City").agg(
                Total_Cases=("Population_Affected","sum"),
                Total_Deaths=("Number_of_Deaths","sum"),
                Total_Survived=("Survived","sum"),
                Avg_CFR=("CFR","mean"),
                Diseases=("Disease","nunique")
            ).reset_index()
            city_summary["Avg_CFR"] = city_summary["Avg_CFR"].round(2)

            st.dataframe(city_summary, use_container_width=True)

            tabs = st.tabs(["📊 Cases", "💀 Deaths", "📉 CFR", "🏥 By Disease"])
            with tabs[0]:
                chart = {
                    "tooltip": {}, "backgroundColor": "transparent",
                    "xAxis": {"type": "category", "data": city_summary["City"].tolist(), "axisLabel": {"color": "#64748b"}},
                    "yAxis": {"type": "value", "axisLabel": {"color": "#64748b"}},
                    "series": [{"type": "bar", "data": city_summary["Total_Cases"].tolist(),
                                "itemStyle": {"color": "#6366f1"},
                                "label": {"show": True, "position": "top", "color": "#e2e8f0"}}]
                }
                render_chart(chart)
            with tabs[1]:
                chart2 = {
                    "tooltip": {}, "backgroundColor": "transparent",
                    "xAxis": {"type": "category", "data": city_summary["City"].tolist(), "axisLabel": {"color": "#64748b"}},
                    "yAxis": {"type": "value", "axisLabel": {"color": "#64748b"}},
                    "series": [{"type": "bar", "data": city_summary["Total_Deaths"].tolist(),
                                "itemStyle": {"color": "#ef4444"},
                                "label": {"show": True, "position": "top", "color": "#e2e8f0"}}]
                }
                render_chart(chart2)
            with tabs[2]:
                chart3 = {
                    "tooltip": {}, "backgroundColor": "transparent",
                    "xAxis": {"type": "category", "data": city_summary["City"].tolist(), "axisLabel": {"color": "#64748b"}},
                    "yAxis": {"type": "value", "name": "CFR %", "axisLabel": {"color": "#64748b"}},
                    "series": [{"type": "bar", "data": city_summary["Avg_CFR"].tolist(),
                                "itemStyle": {"color": "#f59e0b"},
                                "label": {"show": True, "position": "top", "color": "#e2e8f0"}}]
                }
                render_chart(chart3)
            with tabs[3]:
                disease_comp = comp_data.groupby(["City","Disease"])["Population_Affected"].sum().reset_index()
                all_diseases = disease_comp["Disease"].unique().tolist()
                series = []
                palette = ["#00d4aa","#6366f1","#f59e0b","#ef4444","#9333ea","#3b82f6"]
                for i, d in enumerate(all_diseases):
                    dd = disease_comp[disease_comp["Disease"] == d]
                    city_vals = {row["City"]: row["Population_Affected"] for _, row in dd.iterrows()}
                    series.append({
                        "name": d, "type": "bar",
                        "data": [city_vals.get(c, 0) for c in selected],
                        "itemStyle": {"color": palette[i % len(palette)]}
                    })
                chart4 = {
                    "tooltip": {"trigger": "axis"}, "backgroundColor": "transparent",
                    "legend": {"data": all_diseases, "textStyle": {"color": "#e2e8f0"}},
                    "xAxis": {"type": "category", "data": selected, "axisLabel": {"color": "#64748b"}},
                    "yAxis": {"type": "value", "axisLabel": {"color": "#64748b"}},
                    "series": series
                }
                render_chart(chart4)

# ─────────────────────────────────────────────────────────────
# SEASONAL ANALYSIS
# ─────────────────────────────────────────────────────────────
if show_seasonal:
    st.markdown("<div class='section-header'>🍂 Seasonal Disease Pattern Analysis</div>", unsafe_allow_html=True)
    if disease_df.empty:
        st.error("Disease data required.")
    else:
        city_data = disease_df[disease_df["City"].str.lower() == disease_city.lower()]
        if city_data.empty:
            st.error(f"No data for {disease_city}")
        else:
            seasonal = city_data.groupby(["Season","Disease"]).agg(
                Total_Cases=("Population_Affected","sum"),
                Total_Deaths=("Number_of_Deaths","sum"),
                Avg_CFR=("CFR","mean")
            ).reset_index()

            tabs = st.tabs(["📊 Cases by Season", "💀 Deaths by Season", "🔥 Heatmap"])
            seasons_order = ["Winter","Summer","Monsoon","Post-Monsoon"]

            with tabs[0]:
                all_diseases = seasonal["Disease"].unique().tolist()
                palette = ["#00d4aa","#6366f1","#f59e0b","#ef4444","#9333ea"]
                series = []
                for i, d in enumerate(all_diseases):
                    dd = seasonal[seasonal["Disease"] == d]
                    sv = {row["Season"]: row["Total_Cases"] for _, row in dd.iterrows()}
                    series.append({
                        "name": d, "type": "bar",
                        "data": [sv.get(s, 0) for s in seasons_order],
                        "itemStyle": {"color": palette[i % len(palette)]}
                    })
                chart = {
                    "tooltip": {"trigger": "axis"}, "backgroundColor": "transparent",
                    "legend": {"data": all_diseases, "textStyle": {"color": "#e2e8f0"}},
                    "xAxis": {"type": "category", "data": seasons_order, "axisLabel": {"color": "#64748b"}},
                    "yAxis": {"type": "value", "axisLabel": {"color": "#64748b"}},
                    "series": series
                }
                render_chart(chart)

            with tabs[1]:
                series2 = []
                for i, d in enumerate(all_diseases):
                    dd = seasonal[seasonal["Disease"] == d]
                    sv = {row["Season"]: row["Total_Deaths"] for _, row in dd.iterrows()}
                    series2.append({
                        "name": d, "type": "bar",
                        "data": [sv.get(s, 0) for s in seasons_order],
                        "itemStyle": {"color": palette[i % len(palette)]}
                    })
                chart2 = {
                    "tooltip": {"trigger": "axis"}, "backgroundColor": "transparent",
                    "legend": {"data": all_diseases, "textStyle": {"color": "#e2e8f0"}},
                    "xAxis": {"type": "category", "data": seasons_order, "axisLabel": {"color": "#64748b"}},
                    "yAxis": {"type": "value", "axisLabel": {"color": "#64748b"}},
                    "series": series2
                }
                render_chart(chart2)

            with tabs[2]:
                pivot = seasonal.pivot_table(index="Disease", columns="Season", values="Total_Cases", fill_value=0)
                fig, ax = plt.subplots(figsize=(10, 5))
                fig.patch.set_facecolor('#111827')
                ax.set_facecolor('#0a0f1e')
                sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                            ax=ax, linewidths=0.5, linecolor='#1e293b',
                            cbar_kws={'shrink': 0.8})
                ax.set_title(f"Disease Cases by Season — {disease_city}", color='#e2e8f0', pad=12)
                ax.tick_params(colors='#64748b')
                plt.xticks(rotation=30, ha='right')
                st.pyplot(fig)
                plt.close()

# ─────────────────────────────────────────────────────────────
# CORRELATION & RISK
# ─────────────────────────────────────────────────────────────
if show_correlation:
    st.markdown("<div class='section-header'>📊 Environmental–Disease Correlation & Risk Assessment</div>", unsafe_allow_html=True)

    lat, lon = get_coordinates(weather_aqi_city)
    if lat is None:
        st.error(f"Could not get coordinates for {weather_aqi_city}")
    else:
        with st.spinner("Analysing environmental data..."):
            w_data = get_current_weather(weather_aqi_city, api)
            forecast_data = get_weather_data(weather_aqi_city, api)
            aqi_data = get_comprehensive_aqi_data(lat, lon, api)

        if 'error' in w_data:
            st.error(f"Weather error: {w_data.get('error')}")
        else:
            temp        = w_data["main"]["temp"]
            humidity    = w_data["main"]["humidity"]
            rain        = w_data.get("rain", {}).get("1h", 0)
            pressure    = w_data["main"]["pressure"]
            wind_speed  = w_data.get("wind", {}).get("speed", 0)

            temp_change = 0
            if "list" in forecast_data:
                temps = [f["main"]["temp"] for f in forecast_data["list"][:8]]
                if temps: temp_change = max(temps) - min(temps)

            latest_pm25 = latest_pm10 = latest_aqi = 0
            dominant_pollutant = "None"
            if 'error' not in aqi_data:
                comp = aqi_data['components']
                res  = aqi_data['aqi_result']
                latest_pm25 = comp.get('pm2_5', 0)
                latest_pm10 = comp.get('pm10', 0)
                latest_aqi  = res['overall_aqi'] or 0
                dominant_pollutant = res['dominant_pollutant'] or "None"

            # Env summary table
            env_dict = {
                "Temperature (°C)": f"{temp:.1f}",
                "Humidity (%)":     str(humidity),
                "Rainfall (mm/h)":  f"{rain:.1f}",
                "Wind Speed (m/s)": f"{wind_speed:.1f}",
                "Pressure (hPa)":   str(pressure),
                "PM2.5 (µg/m³)":    f"{latest_pm25:.1f}",
                "PM10 (µg/m³)":     f"{latest_pm10:.1f}",
                "AQI":              str(latest_aqi),
                "Dominant Pollutant": dominant_pollutant,
                "Temp Variation (°C)": f"{temp_change:.1f}",
            }
            with st.expander("📋 Current Environmental Conditions", expanded=False):
                st.dataframe(pd.DataFrame(list(env_dict.items()), columns=["Parameter","Value"]),
                             use_container_width=True)

            # Compute risk
            risk_results = compute_disease_risk(
                temp, humidity, rain, wind_speed,
                latest_aqi, latest_pm25, latest_pm10,
                dominant_pollutant, temp_change
            )

            # Display risk cards
            st.markdown("#### 🔬 Disease Risk Assessment")
            cols = st.columns(3)
            for i, r in enumerate(risk_results):
                risk_class = "risk-high" if r["Risk"]=="High" else "risk-mod" if r["Risk"]=="Moderate" else "risk-low"
                color = "#ef4444" if r["Risk"]=="High" else "#f59e0b" if r["Risk"]=="Moderate" else "#00d4aa"
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class='metric-card' style='text-align:left; margin-bottom:10px;'>
                        <div style='font-size:0.85rem; font-family: Space Mono, monospace; color:#e2e8f0; margin-bottom:6px;'>{r["Disease"]}</div>
                        <div style='margin-bottom:6px;'><span class='{risk_class}'>{r["Risk"]} Risk</span></div>
                        <div style='font-size:0.75rem; color:#64748b;'>Score: {r["Score"]}/{r["Max_Score"]} ({r["Risk_Pct"]}%)</div>
                        <div style='background:#1e293b; border-radius:4px; height:4px; margin-top:6px;'>
                            <div style='background:{color}; height:4px; width:{r["Risk_Pct"]}%; border-radius:4px;'></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            corr_df = pd.DataFrame(risk_results)
            color_map = {"Low":"#00d4aa","Moderate":"#f59e0b","High":"#ef4444"}

            tabs = st.tabs(["📊 Risk Chart", "📈 Score Comparison", "🕸 Radar", "🥧 Distribution", "📝 Detail"])

            with tabs[0]:
                bar_data = [{"value": 3 if r=="High" else 2 if r=="Moderate" else 1,
                              "itemStyle": {"color": color_map[r]}}
                            for r in corr_df["Risk"]]
                chart = {
                    "tooltip": {"trigger": "axis"}, "backgroundColor": "transparent",
                    "xAxis": {"type": "category", "data": corr_df["Disease"].tolist(), "axisLabel": {"color": "#64748b", "rotate": 20}},
                    "yAxis": {"type": "value", "max": 3, "axisLabel": {
                        "color": "#64748b",
                        "formatter": "function(v){return ['','Low','Moderate','High'][v];}"
                    }},
                    "series": [{"type": "bar", "data": bar_data,
                                "label": {"show": True, "position": "top", "color": "#e2e8f0",
                                          "formatter": "function(p){return ['','Low','Moderate','High'][p.value];}"}}]
                }
                render_chart(chart)

            with tabs[1]:
                score_data = [{"value": r["Risk_Pct"], "itemStyle": {"color": color_map[r["Risk"]]}}
                              for r in risk_results]
                chart2 = {
                    "tooltip": {"trigger": "axis"}, "backgroundColor": "transparent",
                    "xAxis": {"type": "category", "data": corr_df["Disease"].tolist(), "axisLabel": {"color": "#64748b", "rotate": 20}},
                    "yAxis": {"type": "value", "name": "Risk %", "axisLabel": {"color": "#64748b"}},
                    "series": [{"type": "bar", "data": score_data,
                                "label": {"show": True, "position": "top", "color": "#e2e8f0",
                                          "formatter": "{c}%"}}]
                }
                render_chart(chart2)

            with tabs[2]:
                radar = {
                    "backgroundColor": "transparent",
                    "tooltip": {},
                    "legend": {"data": ["Risk %"], "textStyle": {"color": "#e2e8f0"}},
                    "radar": {"indicator": [{"name": r["Disease"], "max": 100} for r in risk_results],
                              "name": {"textStyle": {"color": "#64748b"}}},
                    "series": [{"name": "Risk Assessment", "type": "radar",
                                "data": [{"value": [r["Risk_Pct"] for r in risk_results], "name": "Risk %",
                                          "lineStyle": {"color": "#00d4aa"},
                                          "areaStyle": {"color": "rgba(0,212,170,0.15)"}}]}]
                }
                render_chart(radar)

            with tabs[3]:
                risk_counts = corr_df["Risk"].value_counts().to_dict()
                pie = {
                    "backgroundColor": "transparent",
                    "tooltip": {"trigger": "item"},
                    "series": [{"type": "pie", "radius": ["40%","70%"],
                                "data": [{"value": v, "name": k, "itemStyle": {"color": color_map[k]}}
                                         for k, v in risk_counts.items()],
                                "label": {"color": "#e2e8f0"}}]
                }
                render_chart(pie, "300px")

            with tabs[4]:
                for r in risk_results:
                    icon = "🔴" if r["Risk"]=="High" else "🟡" if r["Risk"]=="Moderate" else "🟢"
                    alert_class = "alert-high" if r["Risk"]=="High" else "alert-mod" if r["Risk"]=="Moderate" else "alert-low"
                    factors_html = "<br>".join([f"• {f}" for f in r.get("Factors", [])])
                    st.markdown(f"""
                    <div class='{alert_class}'>
                        <strong>{icon} {r["Disease"]}</strong> — {r["Risk"]} Risk ({r["Risk_Pct"]}%)<br>
                        <small style='color:#94a3b8;'>{factors_html}</small>
                    </div>
                    """, unsafe_allow_html=True)

            # Statistical correlation
            if not disease_df.empty:
                st.markdown("<div class='section-header'>📐 Statistical Correlation Analysis</div>", unsafe_allow_html=True)
                city_hist = disease_df[disease_df["City"].str.lower() == disease_city.lower()]
                if not city_hist.empty and len(city_hist) > 3:
                    numeric_cols = ["Population_Affected","Number_of_Deaths","Survived",
                                    "Doctors_Available","Hospitals_Available","CFR"]
                    avail = [c for c in numeric_cols if c in city_hist.columns]
                    corr_matrix = city_hist[avail].corr()

                    fig, ax = plt.subplots(figsize=(8, 6))
                    fig.patch.set_facecolor('#111827')
                    ax.set_facecolor('#0a0f1e')
                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                                cmap="coolwarm", ax=ax, linewidths=0.5,
                                linecolor='#1e293b', vmin=-1, vmax=1,
                                cbar_kws={'shrink': 0.8})
                    ax.set_title(f"Correlation Matrix — {disease_city}", color='#e2e8f0', pad=12)
                    ax.tick_params(colors='#64748b')
                    st.pyplot(fig)
                    plt.close()

            # Recommendations
            st.markdown("<div class='section-header'>🏥 Health Recommendations</div>", unsafe_allow_html=True)
            high_risk = [r for r in risk_results if r["Risk"]=="High"]
            mod_risk  = [r for r in risk_results if r["Risk"]=="Moderate"]
            low_risk  = [r for r in risk_results if r["Risk"]=="Low"]

            recommendations = {
                "Dengue":                "Remove standing water · Use mosquito nets · Wear long sleeves · Use repellents",
                "Heat Stroke":           "Stay indoors 11AM-4PM · Drink water every 20 mins · Light clothing · Use AC/fans",
                "Asthma":                "Limit outdoor activity · Keep inhaler accessible · Use air purifiers · Wear N95 mask",
                "Respiratory Infections":"Boost immunity · Wash hands frequently · Avoid crowds · Stay hydrated",
                "Malaria":               "Use insecticide-treated nets · Eliminate standing water · Seek early treatment",
                "Waterborne Diseases":   "Drink only purified water · Avoid flood water · Maintain hygiene",
            }
            if high_risk:
                st.markdown(f"<div class='alert-high'>🚨 <strong>HIGH ALERT:</strong> {', '.join([r['Disease'] for r in high_risk])}</div>", unsafe_allow_html=True)
                for r in high_risk:
                    rec = recommendations.get(r["Disease"], "Follow standard health guidelines.")
                    st.markdown(f"**{r['Disease']}:** {rec}")
            if mod_risk:
                st.markdown(f"<div class='alert-mod'>⚠️ <strong>MODERATE RISK:</strong> {', '.join([r['Disease'] for r in mod_risk])}</div>", unsafe_allow_html=True)
                st.write("• Monitor symptoms · Take standard preventive measures · Check air quality updates")
            if low_risk:
                st.markdown(f"<div class='alert-low'>✅ <strong>LOW RISK:</strong> {', '.join([r['Disease'] for r in low_risk])}</div>", unsafe_allow_html=True)
                st.write("• Maintain normal activities with basic hygiene practices")

            st.markdown("""<div class='alert-info'><small>⚕️ <strong>Disclaimer:</strong> Risk assessment is based on real-time environmental data and epidemiological models.
            Always consult qualified healthcare professionals for medical decisions.</small></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ML PREDICTION
# ─────────────────────────────────────────────────────────────
if show_ml:
    st.markdown("<div class='section-header'>🤖 Machine Learning — Disease Outbreak Prediction</div>", unsafe_allow_html=True)

    if not ML_AVAILABLE:
        st.error("scikit-learn not installed. Run: pip install scikit-learn")
    elif disease_df.empty:
        st.error("Disease data required for ML.")
    else:
        with st.spinner("Training prediction model..."):
            model, encoders, feature_names, metrics = train_ml_model(disease_df)

        if model is None:
            st.error("Insufficient data to train model (need at least 10 records).")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("🎯 Model", "Gradient Boosting")
            c2.metric("📊 R² Score", f"{metrics['r2']:.3f}")
            c3.metric("📉 MAE", f"{metrics['mae']:.0f} cases")

            st.markdown(f"""<div class='alert-{"low" if metrics["r2"] > 0.7 else "mod" if metrics["r2"] > 0.4 else "high"}'>
            Model accuracy: R² = {metrics['r2']:.3f} &nbsp;|&nbsp; Mean Absolute Error = {metrics['mae']:.0f} cases &nbsp;|&nbsp;
            Test set size = {metrics['test_size']} records
            </div>""", unsafe_allow_html=True)

            st.markdown("#### 🔮 Predict Future Cases")
            col1, col2 = st.columns(2)
            with col1:
                pred_city    = st.selectbox("City", sorted(disease_df["City"].unique()), key="ml_city")
                pred_disease = st.selectbox("Disease", sorted(disease_df["Disease"].unique()), key="ml_disease")
                pred_season  = st.selectbox("Season", ["Winter","Summer","Monsoon","Post-Monsoon"], key="ml_season")
            with col2:
                pred_month    = st.slider("Month", 1, 12, 6, key="ml_month")
                pred_doctors  = st.number_input("Doctors Available", min_value=0, max_value=500, value=50, key="ml_doctors")
                pred_hospitals = st.number_input("Hospitals Available", min_value=0, max_value=100, value=10, key="ml_hospitals")

            if st.button("🔮 Predict Cases", key="ml_predict"):
                try:
                    prediction = predict_cases(model, encoders, feature_names,
                                               pred_city, pred_disease, pred_season,
                                               pred_month, pred_doctors, pred_hospitals)
                    if prediction is not None:
                        st.markdown(f"""
                        <div class='alert-info' style='text-align:center;'>
                            <div style='font-family: Space Mono, monospace; font-size: 2rem; color: #00d4aa;'>{prediction:,}</div>
                            <div style='color:#94a3b8; font-size:0.85rem;'>Predicted cases of <strong>{pred_disease}</strong> in <strong>{pred_city}</strong> during <strong>{pred_season} (Month {pred_month})</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Prediction failed. Check if city/disease exists in training data.")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

            # Feature importance
            if hasattr(model, 'feature_importances_'):
                st.markdown("#### 📊 Feature Importance")
                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": model.feature_importances_
                }).sort_values("Importance", ascending=False)

                fi_chart = {
                    "tooltip": {}, "backgroundColor": "transparent",
                    "xAxis": {"type": "category", "data": importance_df["Feature"].tolist(), "axisLabel": {"color": "#64748b"}},
                    "yAxis": {"type": "value", "axisLabel": {"color": "#64748b"}},
                    "series": [{"type": "bar", "data": importance_df["Importance"].round(4).tolist(),
                                "itemStyle": {"color": "#6366f1"},
                                "label": {"show": True, "position": "top", "color": "#e2e8f0"}}]
                }
                render_chart(fi_chart, "300px")

            # Disease trend prediction
            st.markdown("#### 📈 Multi-Month Forecast")
            forecast_city    = st.selectbox("City for Forecast", sorted(disease_df["City"].unique()), key="fc_city")
            forecast_disease = st.selectbox("Disease for Forecast", sorted(disease_df["Disease"].unique()), key="fc_disease")
            forecast_doctors = st.number_input("Doctors", min_value=0, max_value=500, value=50, key="fc_doctors")
            forecast_hospitals = st.number_input("Hospitals", min_value=0, max_value=100, value=10, key="fc_hospitals")

            if st.button("📈 Generate 12-Month Forecast"):
                season_map = {1:"Winter",2:"Winter",3:"Summer",4:"Summer",5:"Summer",
                              6:"Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
                              10:"Post-Monsoon",11:"Post-Monsoon",12:"Winter"}
                months = list(range(1, 13))
                preds  = []
                for m in months:
                    s = season_map[m]
                    try:
                        p = predict_cases(model, encoders, feature_names,
                                          forecast_city, forecast_disease, s,
                                          m, forecast_doctors, forecast_hospitals)
                        preds.append(p if p is not None else 0)
                    except:
                        preds.append(0)

                month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                fc_chart = {
                    "tooltip": {"trigger": "axis"}, "backgroundColor": "transparent",
                    "xAxis": {"type": "category", "data": month_names, "axisLabel": {"color": "#64748b"}},
                    "yAxis": {"type": "value", "name": "Predicted Cases", "axisLabel": {"color": "#64748b"}},
                    "series": [{"name": "Predicted Cases", "type": "line", "smooth": True,
                                "data": preds,
                                "lineStyle": {"color": "#00d4aa"}, "itemStyle": {"color": "#00d4aa"},
                                "areaStyle": {"color": "rgba(0,212,170,0.15)"},
                                "label": {"show": True, "position": "top", "color": "#e2e8f0"}}]
                }
                render_chart(fc_chart)
                peak_month = month_names[preds.index(max(preds))]
                st.markdown(f"""<div class='alert-info'>
                📅 Peak predicted month: <strong>{peak_month}</strong> with <strong>{max(preds):,}</strong> cases
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# INDIA RISK MAP
# ─────────────────────────────────────────────────────────────
if show_map:
    st.markdown("<div class='section-header'>🗺 India Disease Risk Map</div>", unsafe_allow_html=True)

    if not FOLIUM_AVAILABLE:
        st.warning("Folium not installed. Run: `pip install folium streamlit-folium`")
        st.info("Showing city list as fallback:")
        if not disease_df.empty:
            city_risk = disease_df.groupby("City").agg(
                Total_Cases=("Population_Affected","sum"),
                Total_Deaths=("Number_of_Deaths","sum"),
                Avg_CFR=("CFR","mean")
            ).reset_index().sort_values("Total_Cases", ascending=False)
            st.dataframe(city_risk, use_container_width=True)
    else:
        m = folium.Map(location=[22.5, 78.9], zoom_start=5,
                       tiles='CartoDB dark_matter')

        if not disease_df.empty:
            city_risk = disease_df.groupby("City").agg(
                Total_Cases=("Population_Affected","sum"),
                Total_Deaths=("Number_of_Deaths","sum"),
                Avg_CFR=("CFR","mean")
            ).reset_index()

            all_cities_coords = {**fallback_coords}

            for _, row in city_risk.iterrows():
                city_name = row["City"]
                coords = all_cities_coords.get(city_name.lower())
                if coords is None:
                    cached = get_coordinates(city_name)
                    if cached[0]: coords = cached

                if coords:
                    cases = row["Total_Cases"]
                    deaths = row["Total_Deaths"]
                    cfr = row["Avg_CFR"]
                    radius = max(8, min(30, cases / 500))
                    color = "#ef4444" if cfr > 5 else "#f59e0b" if cfr > 2 else "#00d4aa"

                    folium.CircleMarker(
                        location=coords,
                        radius=radius,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.6,
                        popup=folium.Popup(f"""
                        <div style='font-family:monospace;'>
                            <b>{city_name}</b><br>
                            Cases: {int(cases):,}<br>
                            Deaths: {int(deaths):,}<br>
                            CFR: {cfr:.1f}%
                        </div>""", max_width=200),
                        tooltip=f"{city_name}: {int(cases):,} cases"
                    ).add_to(m)

        st_folium(m, width=900, height=500)

        st.markdown("""
        <div style='font-size:0.75rem; color:#64748b;'>
        🟢 Low CFR (&lt;2%) &nbsp; 🟡 Moderate CFR (2-5%) &nbsp; 🔴 High CFR (&gt;5%) &nbsp; | &nbsp; Circle size = case count
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────
if show_report:
    st.markdown("<div class='section-header'>📄 Download Health Risk Report</div>", unsafe_allow_html=True)

    if not PDF_AVAILABLE:
        st.warning("ReportLab not installed. Run: `pip install reportlab`")
    else:
        st.markdown("Generate a comprehensive PDF health risk report for the selected city.")
        report_city = st.text_input("City for Report", weather_aqi_city, key="report_city")

        if st.button("🖨️ Generate PDF Report"):
            with st.spinner("Generating report..."):
                lat, lon = get_coordinates(report_city)
                w_data = get_current_weather(report_city, api)
                aqi_data_r = get_comprehensive_aqi_data(lat, lon, api) if lat else {'error': 'no coords'}

                temp = humidity = rain = wind_speed = pressure = 0
                latest_pm25 = latest_pm10 = latest_aqi = 0
                dominant_pollutant = "N/A"
                temp_change = 0

                if 'error' not in w_data:
                    temp       = w_data["main"]["temp"]
                    humidity   = w_data["main"]["humidity"]
                    rain       = w_data.get("rain", {}).get("1h", 0)
                    pressure   = w_data["main"]["pressure"]
                    wind_speed = w_data.get("wind", {}).get("speed", 0)

                if 'error' not in aqi_data_r:
                    comp2 = aqi_data_r['components']
                    res2  = aqi_data_r['aqi_result']
                    latest_pm25 = comp2.get('pm2_5', 0)
                    latest_pm10 = comp2.get('pm10', 0)
                    latest_aqi  = res2['overall_aqi'] or 0
                    dominant_pollutant = res2['dominant_pollutant'] or "N/A"

                env_dict_r = {
                    "Temperature (°C)": f"{temp:.1f}",
                    "Humidity (%)": str(humidity),
                    "Rainfall (mm/h)": f"{rain:.1f}",
                    "Wind Speed (m/s)": f"{wind_speed:.1f}",
                    "Pressure (hPa)": str(pressure),
                    "PM2.5 (µg/m³)": f"{latest_pm25:.1f}",
                    "PM10 (µg/m³)": f"{latest_pm10:.1f}",
                    "AQI": str(latest_aqi),
                    "Dominant Pollutant": dominant_pollutant,
                }

                risk_results_r = compute_disease_risk(
                    temp, humidity, rain, wind_speed,
                    latest_aqi, latest_pm25, latest_pm10,
                    dominant_pollutant, temp_change
                )

                pdf_buffer = generate_pdf_report(report_city, risk_results_r, env_dict_r)
                if pdf_buffer:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"EnvHealthIQ_{report_city}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("Report generated successfully!")
                else:
                    st.error("PDF generation failed.")

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
    EnvHealth IQ Dashboard &nbsp;|&nbsp; Built with Streamlit &amp; OpenWeatherMap &nbsp;|&nbsp;
    ML: Gradient Boosting Regressor &nbsp;|&nbsp; AQI: EPA Standard Breakpoints<br>
    <span style='color:#1e2d45;'>⚕️ For academic and informational purposes only. Not a substitute for medical advice.</span>
</div>
""", unsafe_allow_html=True)