# 🌍 EnvHealth IQ — Environment–Disease Intelligence Dashboard

A comprehensive Streamlit dashboard that correlates real-time environmental conditions
with disease outbreak risks across Indian cities.

---

## 🚀 Features

### Original Features (Enhanced)
- **Live Weather Data** — 5-day forecast with temperature, humidity, rainfall, wind charts
- **Comprehensive AQI** — Full EPA-standard AQI calculation for PM2.5, PM10, O₃, NO₂, SO₂, CO
- **Disease Statistics** — Historical case/death/survival data from CSV

### New Features Added
- **🔁 City Comparison** — Side-by-side comparison of 2-4 cities across all disease metrics
- **🍂 Seasonal Analysis** — Disease trend breakdown by Winter/Summer/Monsoon/Post-Monsoon with heatmap
- **📐 Correlation Heatmap** — Pearson correlation matrix between health variables
- **🦠 More Diseases** — Added Malaria and Waterborne Diseases to risk assessment
- **🤖 ML Prediction** — Gradient Boosting model to predict future outbreak case counts
  - 12-month rolling forecast
  - Feature importance visualization
  - Model accuracy metrics (R², MAE)
- **🗺 India Risk Map** — Interactive Folium map with disease burden by city
- **📄 PDF Report** — Downloadable health risk report with full analysis
- **🎨 Dark Dashboard UI** — Professional dark theme with Space Mono typography

---

## 🛠 Setup & Installation

```bash
# 1. Clone or copy the project files
# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenWeatherMap API key
# Option A: Create .streamlit/secrets.toml
echo 'OPENWEATHER_API_KEY = "your_key_here"' > .streamlit/secrets.toml

# Option B: Set environment variable
export OPENWEATHER_API_KEY="your_key_here"

# 4. Make sure your CSV file is in the same directory
# File: output_d206b0_corrected.csv

# 5. Run the app
streamlit run app.py
```

---

## 📁 File Structure

```
env_disease_dashboard/
├── app.py                          ← Main application file
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
├── output_d206b0_corrected.csv     ← Disease dataset (your CSV)
└── .streamlit/
    └── secrets.toml                ← API keys (create this)
```

---

## 📊 CSV Format Required

Your CSV file must have these columns:

| Column | Type | Description |
|--------|------|-------------|
| City | string | City name |
| Disease | string | Disease name |
| Date | date | YYYY-MM-DD format |
| Population_Affected | int | Number of cases |
| Number_of_Deaths | int | Deaths |
| Survived | int | Survived count |
| Doctors_Available | int | Doctors count |
| Hospitals_Available | int | Hospitals count |

---

## 🧬 Diseases Monitored

| Disease | Key Environmental Triggers |
|---------|---------------------------|
| Dengue | Rainfall, Humidity >70%, Temp 25-32°C |
| Heat Stroke | Temp >35°C, High humidity, Low wind |
| Asthma | PM2.5/PM10, AQI, Temperature variation |
| Respiratory Infections | Cold weather, Poor AQI, High humidity |
| Malaria | Temp 20-30°C, Rainfall, High humidity |
| Waterborne Diseases | Heavy rainfall, Warm temperatures |

---

## 🤖 ML Model Details

- **Algorithm**: Gradient Boosting Regressor (scikit-learn)
- **Target**: Population_Affected (case count)
- **Features**: City, Disease, Season, Month, Doctors, Hospitals
- **Split**: 80% train / 20% test
- **Metrics**: R² Score, Mean Absolute Error

---

## 🗝 API Key

Get a free API key at: https://openweathermap.org/api
The free tier supports up to 1000 calls/day which is sufficient for this dashboard.

---

## ⚕️ Disclaimer

This dashboard is built for academic purposes. Risk assessments are based on
environmental data and statistical models. Always consult qualified healthcare
professionals for medical decisions.