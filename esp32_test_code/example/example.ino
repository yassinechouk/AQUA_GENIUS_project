#include "irrigation_ml.h"

void setup() {
    Serial.begin(115200);
    delay(1000);
    Serial.println("\n🚀 ESP32 Irrigation ML System");
    
    // Initialiser ML
    irrigation_ml_init();
}

void loop() {
    // Exemple de données capteur
    float ETo = 5.2;              // mm/jour
    float VPD = 1.8;              // kPa
    float soil_moisture = 0.25;   // volumetric
    float soil_temp = 22.5;       // °C
    int categorie = 2;            // Type culture
    
    // Prédiction
    IrrigationPrediction pred = predict_irrigation(
        ETo, VPD, soil_moisture, soil_temp, categorie
    );
    
    // Affichage
    print_prediction(&pred);
    
    // Attendre 10 secondes
    delay(10000);
}