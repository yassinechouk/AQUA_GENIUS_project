# 🌱 AQUA GENIUS - Smart Irrigation System

<div align="center">


**An intelligent, ML-powered irrigation system optimized for precision agriculture**

[Features](#-features) • [Architecture](#-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Hardware](#-hardware) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

AQUA GENIUS is an end-to-end smart irrigation solution that combines **Machine Learning**, **IoT sensors**, and **embedded systems** to optimize water usage in agriculture. Originally designed for Tunisian farming conditions, it can be adapted to any agricultural context.

The system collects environmental data from multiple sources, trains XGBoost models for irrigation decisions, and deploys them directly on an ESP32-S3 microcontroller for **real-time edge inference** — no cloud required.

---

## ✨ Features

- 🤖 **ML-Powered Decisions** — XGBoost models for pump control (ON/OFF) and water volume prediction
- 📡 **Multi-Source Data Pipeline** — Integrates NASA POWER, CIMIS, and synthetic datasets
- ⚡ **Edge Deployment** — Models run directly on ESP32-S3 for low-latency decisions
- 📱 **Mobile Control** — Blynk app for remote monitoring and manual override
- 🔧 **Auto-Calibration** — Sensor calibration and error detection built-in
- 💧 **Water Optimization** — Reduces water consumption through data-driven irrigation

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION                          │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   NASA POWER    │     CIMIS       │    Synthetic (Kaggle)       │
│   (Primary)     │   (Optional)    │      (Optional)             │
└────────┬────────┴────────┬────────┴──────────────┬──────────────┘
         │                 │                       │
         └─────────────────┼───────────────────────┘
                           ▼
              ┌────────────────────────┐
              │   merge_all_sources. py │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │generate_final_dataset. py│
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │       train. py         │
              │  (XGBoost Training)    │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │xgboost_esp32_converter │
              │   (. pkl → C/C++)       │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │      ESP32-S3          │
              │  (Edge Inference)      │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │    Blynk IoT App       │
              │ (Monitoring & Control) │
              └────────────────────────┘
```

---

## 📁 Project Structure

```
AQUA_GENIUS_project/
│
├── 📊 Data Collection
│   ├── collect_nasa_power. py      # NASA POWER API data collection
│   ├── collect_cimis. py           # CIMIS weather station data
│   ├── collect_kaggle. py          # Synthetic dataset generation
│   └── merge_all_sources. py       # Data fusion script
│
├── 🤖 Machine Learning
│   ├── generate_final_dataset. py  # Dataset cleaning & preparation
│   ├── train.py                   # XGBoost model training
│   ├── test_models.py             # Model validation & testing
│   └── xgboost_esp32_converter.py # Convert . pkl to C/C++
│
├── 📦 models_esp32/               # Trained models (. pkl files)
│
├── 🔌 esp32_test_code/
│   ├── converted_models/          # C/C++ model files (. h, .cpp)
│   ├── examples/                  # Arduino test sketches
│   └── README.md                  # ESP32 setup instructions
│
├── 📱 blynk_akwa_wehd/            # Blynk IoT integration code
│
└── 🧪 codetest/                   # Motor control & sensor tests
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Arduino IDE or PlatformIO
- ESP32-S3 board
- Blynk account (free tier works)

### 1. Clone the Repository

```bash
git clone https://github. com/yourusername/AQUA_GENIUS_project.git
cd AQUA_GENIUS_project
```

### 2.  Install Python Dependencies

```bash
pip install -r requirements.txt
```

<details>
<summary>📦 Required packages</summary>

```
pandas
numpy
scikit-learn
xgboost
requests
pickle
```
</details>

### 3.  Collect Data

```bash
# Collect from NASA POWER (primary source)
python collect_nasa_power. py

# Optional: Collect from CIMIS
python collect_cimis.py

# Optional: Generate synthetic data
python collect_kaggle.py

# Merge all sources
python merge_all_sources.py
```

### 4. Train Models

```bash
# Generate final clean dataset
python generate_final_dataset.py

# Train XGBoost models
python train.py

# Test model performance
python test_models.py
```

### 5.  Deploy to ESP32

```bash
# Convert models to C/C++
python xgboost_esp32_converter.py
```

Then upload the generated files to your ESP32-S3 using Arduino IDE. 

---

## 📊 Data Variables

| Variable | Description | Source |
|----------|-------------|--------|
| `tmean`, `tmin`, `tmax` | Air temperature (°C) | NASA, CIMIS |
| `humidite` | Air humidity (%) | NASA, CIMIS |
| `Ra` | Solar radiation (MJ/m²/day) | Calculated |
| `ETo` | Evapotranspiration (mm/day) | NASA, CIMIS |
| `VPD` | Vapor Pressure Deficit (kPa) | Calculated |
| `soil_temp` | Soil temperature (°C) | CIMIS |
| `soil_moisture` | Soil humidity (%) | CIMIS |

---

## 🔌 Hardware

### Components Required

| Component | Purpose |
|-----------|---------|
| ESP32-S3 | Main microcontroller |
| Soil moisture sensor | Ground humidity measurement |
| DHT22 / BME280 | Air temperature & humidity |
| Ultrasonic sensor (HC-SR04) | Water level / safety detection |
| Relay module | Pump control |
| Water pump | Irrigation |

### Wiring Diagram

```
ESP32-S3
    │
    ├── GPIO XX ──► Soil Moisture Sensor
    ├── GPIO XX ──► DHT22 (Temp/Humidity)
    ├── GPIO XX ──► HC-SR04 TRIG
    ├── GPIO XX ──► HC-SR04 ECHO
    ├── GPIO XX ──► Relay IN (Pump Control)
    └── WiFi ──────► Blynk Cloud
```

> 📌 See `esp32_test_code/README.md` for detailed pin configurations. 

---

## 📱 Blynk App Features

| Feature | Description |
|---------|-------------|
| 🟢 Auto Mode | ML-based automatic irrigation |
| 🔵 Manual Mode | Direct pump ON/OFF control |
| 📊 Dashboard | Real-time sensor readings |
| ⚠️ Alerts | WiFi, API, and sensor failure notifications |
| 📈 History | Irrigation logs and water usage stats |

---

## 🧠 Machine Learning Models

### Classification: Pump Status
- **Task:** Predict pump ON (1) or OFF (0)
- **Algorithm:** XGBoost Classifier
- **Features:** Temperature, humidity, soil moisture, ETo, VPD

### Regression: Irrigation Volume
- **Task:** Predict water volume needed (mm/day)
- **Algorithm:** XGBoost Regressor
- **Features:** Same as classifier + crop type, surface area

---

## 🧪 Testing

```bash
# Test trained models with sample inputs
python test_models.py
```

Example output:
```
🔍 Test Results:
├── Pump Status: ON (confidence: 94. 2%)
└── Irrigation Volume: 4.7 mm/day
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

---

## 👥 Authors

- **Yassine Chouk** - *Initial work* - [@yassinechouk](https://github.com/yassinechouk)

---

## 🙏 Acknowledgments

- NASA POWER API for meteorological data
- California CIMIS for irrigation sensor data
- XGBoost team for the ML library
- Blynk for IoT platform

---

<div align="center">

**⭐ Star this repo if you found it helpful!**

Made with 💧 for sustainable agriculture

</div>
