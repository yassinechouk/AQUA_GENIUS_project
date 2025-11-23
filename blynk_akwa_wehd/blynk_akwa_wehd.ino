#define BLYNK_TEMPLATE_ID "TMPL22-Uhe5pP"
#define BLYNK_TEMPLATE_NAME "Aquagenius"
#define BLYNK_AUTH_TOKEN "avirSD78yp7GuTKwhRqy_9dHJDDflWhc"
#include "irrigation_ml.h"
#include <WiFi.h>
#include <BlynkSimpleEsp32.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <OneWire.h>
#include <DallasTemperature.h>
const int PIN_BOUTONV7 = 21;
const int PIN_modele = 22;  
int sortiemodele=0;
float volume=0;
// ================= CONFIGURATION WIFI =================
const char* ssid = "Redmi Note 12 Pro";
const char* pass = "1234567890";

// ================= CAPTEURS =================
#define HUMIDITY_PIN 18
#define TEMP_PIN 2   // broche DATA du DS18B20

OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);

// ================= VARIABLES =================
float localTemperature = 0;  // DS18B20
float localHumidity = 0;     // capteur analogique
int categorie = 0;                // V3 converti 1-2-3
int surface = 0;             // V4
int boutonV7 = 0;            // V7 marche/arrêt
BlynkTimer timer;

// ================= METEO OPENWEATHER =================
float lat = 36.8625;
float lon = 10.1956;
String apiKey = "76ef30eb1fae7a1b33ad5d703ec438d6";

struct WeatherData {
  float tempMoyenne;
  int humidite;
  
  String description;
};
WeatherData weather;
BLYNK_WRITE(V7)
{
  boutonV7 = param.asInt();   // 1 = ON, 0 = OFF

  Serial.print("Bouton V7 = ");
  Serial.println(boutonV7);

  if (boutonV7 == 1) digitalWrite(PIN_BOUTONV7, HIGH);
  else digitalWrite(PIN_BOUTONV7, LOW);
  
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  delay(1000);
  irrigation_ml_init();
  pinMode(PIN_BOUTONV7, OUTPUT);
  digitalWrite(PIN_BOUTONV7, LOW);
  // WiFi et Blynk
  WiFi.begin(ssid, pass);
  Serial.print("Connexion Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWi-Fi connecté !");
  Blynk.begin(BLYNK_AUTH_TOKEN, ssid, pass);

  // Initialisation DS18B20
  sensors.begin();

  // Timers
  timer.setInterval(2000L, sendLocalSensors);  // capteurs locaux
  timer.setInterval(10000L, updateWeather);    // météo OpenWeather
}

// ================= LOOP =================
void loop() {
  Blynk.run();
  timer.run();
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 2000)
  {
    Serial.print("Valeur actuelle boutonV7 = ");
    Serial.println(boutonV7);
    lastPrint = millis();
  }
  float ETo = 8.84;              // mm/jour
  float VPD = 0.59;              // kPa
  IrrigationPrediction pred = predict_irrigation(
        ETo, VPD, localHumidity, localTemperature, categorie
    );
    
    // Affichage
    print_prediction(&pred);
    sortiemodele= pred.pump_status;
    volume= pred.irrigation_volume;
    if (sortiemodele == 1) digitalWrite(PIN_modele, HIGH);
    else digitalWrite(PIN_modele, LOW);
    
    // Attendre 10 secondes
    delay(10000);

}

// ================= FONCTIONS =================

// Lecture capteurs locaux
void sendLocalSensors() {
  // Humidité
  int rawValue = analogRead(HUMIDITY_PIN);
  localHumidity = 100 - (rawValue / 4095.0) * 100.0;
  Serial.print("Humidité locale : "); Serial.println(localHumidity);
  Blynk.virtualWrite(V0, localHumidity);

  // Température DS18B20
  sensors.requestTemperatures();
  localTemperature = sensors.getTempCByIndex(0);
  if (localTemperature == DEVICE_DISCONNECTED_C) localTemperature = -999;
  Serial.print("Température locale : "); Serial.println(localTemperature);
  Blynk.virtualWrite(V1, localTemperature);

  // Affichage debug
  Serial.print("categorie(V3 converti) = "); Serial.print(categorie);
  Serial.print(" | surface(V4) = "); Serial.println(surface);
}

// Lecture météo OpenWeatherMap
void updateWeather() {
  if (WiFi.status() != WL_CONNECTED) { WiFi.reconnect(); return; }

  HTTPClient http;
  String url = "http://api.openweathermap.org/data/2.5/weather?lat=" +
               String(lat, 4) + "&lon=" + String(lon, 4) +
               "&appid=" + apiKey + "&units=metric&lang=fr";
  http.begin(url);
  int httpCode = http.GET();
  if (httpCode != 200) { Serial.println("Erreur HTTP"); http.end(); return; }
  String payload = http.getString();
  http.end();

  DynamicJsonDocument doc(1024);
  DeserializationError error = deserializeJson(doc, payload);
  if (error) { Serial.println("Erreur JSON"); return; }

  weather.tempMoyenne = doc["main"]["temp"];
  weather.humidite = doc["main"]["humidity"];
  weather.description = doc["weather"][0]["description"].as<String>();

  Serial.print("Temp moyenne API : "); Serial.println(weather.tempMoyenne);
  Serial.print("Humidité API : "); Serial.println(weather.humidite);
  Serial.print("Description API : "); Serial.println(weather.description);

  // Envoi sur Blynk
  Blynk.virtualWrite(V2, weather.tempMoyenne);
  Blynk.virtualWrite(V6, weather.humidite);
  Blynk.virtualWrite(V5, weather.description);
}

// ================= BLYNK WRITE =================
BLYNK_WRITE(V3) {
  int valeur = param.asInt();
  if (valeur >= 0 && valeur <= 2) categorie = 1;
  else if (valeur >= 3 && valeur <= 5) categorie = 2;
  else if (valeur >= 6 && valeur <= 8) categorie = 3;
  else categorie = 0;
  Serial.print("V3 = "); Serial.print(valeur); Serial.print(" -> categorie = "); Serial.println(categorie);
}

BLYNK_WRITE(V4) {
  surface = param.asInt();
  Serial.print("V4 = "); Serial.println(surface);
}
