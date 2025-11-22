/*
 * Irrigation ML Models for ESP32-S3
 * Auto-généré le: 2025-11-22 18:41:25
 * Auteur: Yassinechouk
 * 
 * FEATURES (5 variables):
 *   [0] ETo
 *   [1] VPD
 *   [2] soil_moisture
 *   [3] soil_temp
 *   [4] categorie
 * 
 * OUTPUTS:
 *   - pump_status: 0=OFF, 1=ON
 *   - irrigation_volume: mm/jour (0-15)
 */

#ifndef IRRIGATION_ML_H
#define IRRIGATION_ML_H

#include <Arduino.h>

// Index des features
enum FeatureIndex {
    FEAT_ETO = 0,
    FEAT_VPD = 1,
    FEAT_SOIL_MOISTURE = 2,
    FEAT_SOIL_TEMP = 3,
    FEAT_CATEGORIE = 4,
};

// Résultat de prédiction
typedef struct {
    int pump_status;           // 0=OFF, 1=ON
    float irrigation_volume;   // mm/jour
    float confidence;          // 0.0-1.0
} IrrigationPrediction;

// Fonctions principales
void irrigation_ml_init();
IrrigationPrediction predict_irrigation(float ETo, float VPD, float soil_moisture, float soil_temp, int categorie);
void print_prediction(const IrrigationPrediction* pred);

#endif // IRRIGATION_ML_H