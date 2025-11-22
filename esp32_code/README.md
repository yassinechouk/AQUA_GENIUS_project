# Irrigation ML Models for ESP32-S3

## 📋 Description
Modèles XGBoost convertis en C pour prédiction d'irrigation sur ESP32-S3.

**Performances identiques au modèle Python original!**

## 🎯 Features
- **5 inputs**: ETo, VPD, soil_moisture, soil_temp, categorie
- **2 outputs**: pump_status (0/1), irrigation_volume (mm/jour)
- **Optimisé**: Compact, rapide, faible consommation mémoire

## 📦 Installation

1. Copier les fichiers dans votre projet Arduino:
   - `irrigation_ml.h`
   - `irrigation_ml.cpp`

2. Inclure dans votre sketch:
```cpp
#include "irrigation_ml.h"
```

## 🚀 Utilisation

```cpp
void setup() {
    Serial.begin(115200);
    irrigation_ml_init();
}

void loop() {
    // Lire capteurs
    float ETo = read_eto();
    float VPD = read_vpd();
    float soil_moisture = read_soil_moisture();
    float soil_temp = read_soil_temp();
    int categorie = 2;
    
    // Prédiction
    IrrigationPrediction pred = predict_irrigation(
        ETo, VPD, soil_moisture, soil_temp, categorie
    );
    
    // Utiliser résultat
    if (pred.pump_status == 1) {
        start_pump(pred.irrigation_volume);
    }
    
    delay(60000);  // 1 minute
}
```

## 📊 Performance
- **Modèle classifier**: 100 arbres, profondeur 4
- **Modèle regressor**: 100 arbres, profondeur 4
- **Mémoire Flash**: ~400KB
- **RAM**: ~1KB pendant exécution
- **Temps prédiction**: <10ms

## ✅ Validation
Les prédictions sont **identiques** au modèle Python original (testées).

## 📝 Notes
- Nécessite ESP32-S3 (ou ESP32 avec suffisamment de mémoire)
- Compatible Arduino IDE et PlatformIO
- Pas de dépendances externes

## 👤 Auteur
Yassinechouk

Généré le: 2025-11-22 18:41:25
