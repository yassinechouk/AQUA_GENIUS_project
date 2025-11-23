# AQUA_GENIUS_project
ICI vous pouvez consulter la partie software de notre projet
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
collect_cimis.py--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Script Python pour collecter et traiter les données météorologiques et agronomiques depuis le système CIMIS (California Irrigation Management Information System), adapté pour l'agriculture tunisienne.
variable:
✅ tmean, tmin, tmax → Températures
✅ humidite → Humidité air
✅ Ra → Radiation (calculée)
✅ ETo → Évapotranspiration
✅ VPD → Stress hydrique (calculé)
✅ soil_temp → Température sol
✅ soil_moisture → Humidité sol
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
collect_kaggle.py--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Script Python pour générer des données synthétiques réalistes de sol, cultures et irrigation adaptées au contexte agricole tunisien. Simule un dataset type "Kaggle" pour l'entraînement de modèles de Machine Learning en irrigation.
date | tmin | tmax | tmean | humidite | Ra | VPD | ETo
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
collect_nasa_power.py--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Ce script collecte les données météorologiques et environnementales journalières depuis l’API NASA POWER, puis il :
📥 Télécharge les variables brutes (température, humidité, etc.),
🔢 Calcule certaines variables dérivées (radiation, évapotranspiration, VPD),
🧹 Nettoie et formate les données,
💾 Sauvegarde le résultat dans un fichier CSV prêt pour l’analyse ou l’utilisation dans ton projet AI/IoT.
variables :
Rain|ETo|HumAir|HumSol|Temp|Month|Day|Heure
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
merge_all_sources.py--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Ce script fusionne automatiquement des données provenant de trois sources différentes :
📡 NASA POWER (source principale)
📊 Kaggle (optionnelle)
💧 CIMIS (optionnelle, capteurs irrigations Californie)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
generate_final_dataset.py--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Le script génère automatiquement un dataset final parfait(netoyage et filtrage de données...) pour l’irrigation intelligente, à partir d’un dataset pré-fusionné (NASA + Kaggle + CIMIS).
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
train.py--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Ce code Python est un script complet pour entraîner et préparer des modèles XGBoost optimisés pour une carte ESP32-S3.
Charge un dataset CSV.
Nettoie et vérifie les données.
Normalise et split train/test.
Entraîne un modèle de classification (pump_status).
Entraîne un modèle de régression (irrigation_volume).
Sauvegarde les modèles et le scaler pour ESP32.
Affiche les métriques et importance des features.
Fournit un pipeline prêt pour déployer sur ESP32-S3.

lenear regression -->

XGBOOST ???
XGBoost est une bibliothèque open-source de machine learning très populaire pour les tâches de classification et de régression, basée sur les arbres de décision. Le nom « XGBoost » vient de Extreme Gradient Boosting.
Voici une explication claire et détaillée :
1. Principe de base
XGBoost est un algorithme de boosting, ce qui signifie qu’il combine plusieurs arbres de décision faibles pour créer un modèle puissant.
Un arbre de décision simple est souvent faible (peu précis).
Le boosting entraîne les arbres séquentiellement, chaque nouvel arbre essayant de corriger les erreurs des arbres précédents.
XGBoost utilise le gradient de la fonction de perte pour optimiser les arbres, d’où le nom Gradient Boosting.
classification--> irrigation ON/OFF
lenear regression --> VOLume d'eau necessaire our l'irrigation
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_models.py--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Le script charge les modèles XGBoost entraînés pour un ESP32-S3 et permet de tester leurs prédictions dans différents contextes :
Classification : pump_status (0 = OFF, 1 = ON)
Régression : irrigation_volume (mm/jour)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
xgboost_esp32_converter.py--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Le script convertit des modèles XGBoost entraînés en Python (.pkl) en code C/C++ optimisé pour ESP32-S3, permettant de les utiliser directement sur une carte Arduino/ESP32 sans dépendances Python.
Il gère :
Un classifier (pump_status) → ON/OFF pompe
Un regressor (irrigation_volume) → volume d’irrigation en mm/jour
Un scaler (normalisation des features)
Génération de fichiers pour Arduino : .h, .cpp
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
models_esp32--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
ce dossier contient tout les fichier pkl generer par le code "train.py"
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
esp32_test_code
dans ce dossier il ya un dossier qui contient les conversion des fichier .pkl en c++ et c;
aussi un code dans le dossier exemple pour tester le model machine learning generé, il ya un READ_ME DANS CE DOSIER pour plus d information .
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
blynk_akwa_wehd-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🟦 1. Fonction complète d’arrosage automatique
Active la pompe selon la prédiction
Intègre les capteurs + météo + modèle ML
🟩 2. Fonction d’arrosage manuel via Blynk
Bouton ON/OFF dans l’app
Protection sécurité
🟧 3. Fonction de calcul intelligent du volume d’eau
Basée sur météo, surface, catégorie, humidité…
🟨 4. Fonction de calibrage capteur d’humidité
Conversion correcte 0–100%
Courbe de calibration
🟪 5. Fonction de détection d’erreurs
WiFi OFF
API non disponible
Capteurs débranchés

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
codetest-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Contrôler deux moteurs M1 et M2 en fonction d’un capteur ultrason et de commandes externes pour la sécurité ou override manuel.
En étapes fonctionnelles :
Initialisation
Configure le capteur ultrason (TRIG/ECHO)
Configure les moteurs M1 et M2
Configure les pins de commande externe M2
Stoppe tous les moteurs au démarrage
Mesure de distance
Fonction mesurer_distance() renvoie la distance en cm vers un obstacle
Vérifie les limites (2–400 cm), sinon stoppe les moteurs
Lecture commandes externes M2
Si commandeM2_1 ou commandeM2_2 = HIGH → M2 tourne immédiatement (override)
Ignore la logique ultrason si override actif
Décision automatique basée sur la distance
Si distance > seuil → M1 tourne (pompage ou action 1)
Si distance ≤ seuil → M2 tourne (pompage ou action 2)
Les moteurs ne tournent jamais en même temps, sauf override manuel
Fonctions moteurs
moteurAvance(IN1, IN2, vitesse) → fait tourner un moteur dans un sens avec PWM
moteurStop(IN1, IN2) → stoppe le moteur
PWM permet de contrôler la vitesse de rotation
Boucle principale (loop)
Lit distance et commandes externes
Applique la logique automatique
Affiche tout sur Serial pour debug
Boucle toutes les 300 ms
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






