#!/usr/bin/env python3
"""
Script de Test des Modèles ESP32
Auteur: Yassinechouk
Date: 2025-11-22

Teste les modèles entraînés avec différents scénarios réalistes
"""

import joblib
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelTester:
    """Testeur de modèles d'irrigation"""
    
    def __init__(self, models_dir='models_esp32'):
        self.models_dir = models_dir
        self.clf = None
        self.reg = None
        self.scaler = None
        self.metadata = None
        self.features = ['ETo', 'VPD', 'soil_moisture', 'soil_temp', 'categorie']
        
    def load_models(self):
        """Charge les modèles"""
        print("="*70)
        print("📂 CHARGEMENT DES MODÈLES")
        print("="*70)
        
        clf_path = os.path.join(self.models_dir, 'esp32_pump_classifier.pkl')
        reg_path = os.path.join(self.models_dir, 'esp32_volume_regressor.pkl')
        scaler_path = os.path.join(self.models_dir, 'esp32_scaler.pkl')
        meta_path = os.path.join(self.models_dir, 'esp32_metadata.json')
        
        # Vérifier existence
        for path in [clf_path, reg_path, scaler_path, meta_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Fichier manquant: {path}")
        
        # Charger
        self.clf = joblib.load(clf_path)
        self.reg = joblib.load(reg_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"  ✅ Classificateur chargé")
        print(f"  ✅ Régresseur chargé")
        print(f"  ✅ Scaler chargé")
        print(f"  ✅ Métadonnées chargées")
        
        # Afficher info modèles
        print(f"\n  📊 Informations:")
        print(f"    • Date entraînement: {self.metadata.get('date', 'N/A')}")
        print(f"    • Taille dataset:    {self.metadata.get('n_total', 'N/A')} lignes")
        print(f"    • Features:          {', '.join(self.features)}")
        print()
        
        return self
    
    def predict(self, ETo, VPD, soil_moisture, soil_temp, categorie):
        """Fait une prédiction"""
        
        # Préparer les features
        features = np.array([[ETo, VPD, soil_moisture, soil_temp, categorie]])
        
        # Normaliser
        features_scaled = self.scaler.transform(features)
        
        # Prédictions
        pump_status = self.clf.predict(features_scaled)[0]
        pump_proba = self.clf.predict_proba(features_scaled)[0]
        
        volume_raw = self.reg.predict(features_scaled)[0]
        
        # Cohérence: si pompe OFF → volume = 0
        if pump_status == 0:
            volume = 0.0
        else:
            volume = np.clip(volume_raw, 0, 15)
        
        return {
            'pump_status': bool(pump_status),
            'pump_confidence': float(pump_proba[1] if pump_status == 1 else pump_proba[0]),
            'irrigation_volume': float(volume),
            'volume_raw': float(volume_raw)
        }
    
    def test_scenario(self, name, ETo, VPD, soil_moisture, soil_temp, categorie, description=""):
        """Teste un scénario"""
        
        print(f"\n{'='*70}")
        print(f"📋 SCÉNARIO: {name}")
        print(f"{'='*70}")
        
        if description:
            print(f"📝 {description}")
            print()
        
        # Afficher inputs
        print("📊 DONNÉES CAPTEURS:")
        print(f"  • ETo (Evapotranspiration): {ETo:.2f} mm/jour")
        print(f"  • VPD (Vapor Pressure):     {VPD:.3f} kPa")
        print(f"  • Soil Moisture:            {soil_moisture:.1f} %")
        print(f"  • Soil Temperature:         {soil_temp:.1f} °C")
        print(f"  • Catégorie culture:        {categorie} ({self._get_category_name(categorie)})")
        
        # Prédiction
        result = self.predict(ETo, VPD, soil_moisture, soil_temp, categorie)
        
        # Afficher résultats
        print(f"\n🤖 DÉCISION IA:")
        
        # Status pompe
        status_icon = "✅ ON" if result['pump_status'] else "❌ OFF"
        confidence_pct = result['pump_confidence'] * 100
        print(f"  • Pompe:       {status_icon}")
        print(f"  • Confiance:   {confidence_pct:.1f}%")
        
        # Volume
        if result['pump_status']:
            print(f"  • Volume:      {result['irrigation_volume']:.2f} mm/jour")
            print(f"  • Volume brut: {result['volume_raw']:.2f} mm/jour")
            
            # Durée estimée (exemple: 1mm = 10 minutes)
            duration_min = result['irrigation_volume'] * 10
            print(f"  • Durée:       ~{duration_min:.0f} minutes")
        else:
            print(f"  • Volume:      0.00 mm/jour (pompe OFF)")
        
        # Interprétation
        print(f"\n💡 INTERPRÉTATION:")
        self._interpret_decision(result, ETo, VPD, soil_moisture, soil_temp)
        
        return result
    
    def _get_category_name(self, cat):
        """Retourne le nom de la catégorie"""
        names = {1: "Légumes", 2: "Arbres fruitiers", 3: "Céréales"}
        return names.get(cat, "Inconnu")
    
    def _interpret_decision(self, result, ETo, VPD, soil_moisture, soil_temp):
        """Interprète la décision"""
        
        interpretations = []
        
        # ETo
        if ETo > 4.0:
            interpretations.append("  🌡️  ETo élevé → Forte évapotranspiration")
        elif ETo < 2.0:
            interpretations.append("  🌡️  ETo faible → Faible évapotranspiration")
        
        # VPD
        if VPD > 0.35:
            interpretations.append("  💨 VPD élevé → Air très sec, stress hydrique potentiel")
        elif VPD < 0.15:
            interpretations.append("  💨 VPD faible → Air humide, faible demande évaporative")
        
        # Humidité sol
        if soil_moisture < 45:
            interpretations.append("  🏜️  Sol sec → Irrigation recommandée")
        elif soil_moisture > 65:
            interpretations.append("  💧 Sol très humide → Risque de sur-irrigation")
        elif 50 <= soil_moisture <= 60:
            interpretations.append("  ✅ Humidité sol optimale")
        
        # Température sol
        if soil_temp < 8:
            interpretations.append("  ❄️  Sol froid → Croissance ralentie")
        elif soil_temp > 18:
            interpretations.append("  🔥 Sol chaud → Évaporation accrue")
        
        # Décision globale
        if result['pump_status']:
            if result['irrigation_volume'] > 8:
                interpretations.append("  💧 IRRIGATION IMPORTANTE nécessaire")
            elif result['irrigation_volume'] > 4:
                interpretations.append("  💧 Irrigation modérée recommandée")
            else:
                interpretations.append("  💧 Légère irrigation suffisante")
        else:
            interpretations.append("  ✅ Conditions satisfaisantes, pas d'irrigation")
        
        for interp in interpretations:
            print(interp)
    
    def test_all_scenarios(self):
        """Teste tous les scénarios prédéfinis"""
        
        print("\n" + "="*70)
        print("🚀 TEST DES MODÈLES - SCÉNARIOS RÉALISTES")
        print("="*70)
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
        
        scenarios = [
            {
                'name': "Journée chaude et sèche - Été",
                'description': "Été méditerranéen, sol commençant à sécher",
                'ETo': 5.2,
                'VPD': 0.42,
                'soil_moisture': 48.0,
                'soil_temp': 16.5,
                'categorie': 2
            },
            {
                'name': "Conditions optimales - Printemps",
                'description': "Printemps tempéré, sol bien hydraté",
                'ETo': 2.8,
                'VPD': 0.22,
                'soil_moisture': 58.0,
                'soil_temp': 12.0,
                'categorie': 2
            },
            {
                'name': "Sécheresse sévère",
                'description': "Canicule, sol très sec",
                'ETo': 6.5,
                'VPD': 0.55,
                'soil_moisture': 35.0,
                'soil_temp': 20.0,
                'categorie': 2
            },
            {
                'name': "Après pluie abondante",
                'description': "Sol saturé après précipitations",
                'ETo': 1.5,
                'VPD': 0.12,
                'soil_moisture': 70.0,
                'soil_temp': 10.0,
                'categorie': 2
            },
            {
                'name': "Hiver froid",
                'description': "Hiver, températures basses",
                'ETo': 1.2,
                'VPD': 0.15,
                'soil_moisture': 62.0,
                'soil_temp': 5.0,
                'categorie': 2
            },
            {
                'name': "Légumes en été - Sol moyen",
                'description': "Culture maraîchère, conditions moyennes",
                'ETo': 4.0,
                'VPD': 0.30,
                'soil_moisture': 52.0,
                'soil_temp': 15.0,
                'categorie': 1
            },
            {
                'name': "Céréales début saison",
                'description': "Blé au stade végétatif, printemps",
                'ETo': 2.5,
                'VPD': 0.20,
                'soil_moisture': 55.0,
                'soil_temp': 11.0,
                'categorie': 3
            },
            {
                'name': "Test limites - ETo extrême",
                'description': "Conditions extrêmes de désert",
                'ETo': 8.0,
                'VPD': 0.65,
                'soil_moisture': 30.0,
                'soil_temp': 25.0,
                'categorie': 1
            }
        ]
        
        results = []
        
        for scenario in scenarios:
            result = self.test_scenario(
                name=scenario['name'],
                description=scenario['description'],
                ETo=scenario['ETo'],
                VPD=scenario['VPD'],
                soil_moisture=scenario['soil_moisture'],
                soil_temp=scenario['soil_temp'],
                categorie=scenario['categorie']
            )
            results.append({
                'scenario': scenario['name'],
                'pump': result['pump_status'],
                'volume': result['irrigation_volume'],
                'confidence': result['pump_confidence']
            })
        
        # Résumé
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results):
        """Affiche un résumé des tests"""
        
        print("\n" + "="*70)
        print("📊 RÉSUMÉ DES TESTS")
        print("="*70)
        
        # Statistiques
        n_total = len(results)
        n_on = sum(1 for r in results if r['pump'])
        n_off = n_total - n_on
        
        volumes = [r['volume'] for r in results if r['pump']]
        avg_volume = np.mean(volumes) if volumes else 0
        max_volume = max(volumes) if volumes else 0
        
        print(f"\n  📈 Statistiques:")
        print(f"    • Scénarios testés:    {n_total}")
        print(f"    • Pompe ON:            {n_on} ({n_on/n_total*100:.1f}%)")
        print(f"    • Pompe OFF:           {n_off} ({n_off/n_total*100:.1f}%)")
        
        if volumes:
            print(f"\n  💧 Volumes d'irrigation:")
            print(f"    • Moyen:               {avg_volume:.2f} mm/jour")
            print(f"    • Maximum:             {max_volume:.2f} mm/jour")
        
        # Tableau récapitulatif
        print(f"\n  📋 Tableau récapitulatif:")
        print(f"  {'Scénario':<40} {'Pompe':<10} {'Volume (mm)':<12} {'Conf. %':<10}")
        print(f"  {'-'*72}")
        
        for r in results:
            pump_str = "✅ ON" if r['pump'] else "❌ OFF"
            vol_str = f"{r['volume']:.2f}" if r['pump'] else "-"
            conf_str = f"{r['confidence']*100:.1f}"
            
            print(f"  {r['scenario']:<40} {pump_str:<10} {vol_str:<12} {conf_str:<10}")
        
        print("\n" + "="*70)
        print("✅ TESTS TERMINÉS AVEC SUCCÈS!")
        print("="*70)
    
    def test_custom(self):
        """Test personnalisé interactif"""
        
        print("\n" + "="*70)
        print("🎮 MODE TEST PERSONNALISÉ")
        print("="*70)
        
        print("\nEntrez les valeurs des capteurs:\n")
        
        try:
            ETo = float(input("  ETo (Evapotranspiration mm/jour, ex: 3.5): "))
            VPD = float(input("  VPD (Vapor Pressure kPa, ex: 0.28): "))
            soil_moisture = float(input("  Humidité sol (%, ex: 55): "))
            soil_temp = float(input("  Température sol (°C, ex: 12): "))
            
            print("\n  Catégorie culture:")
            print("    1 = Légumes")
            print("    2 = Arbres fruitiers")
            print("    3 = Céréales")
            categorie = int(input("  Choix (1/2/3): "))
            
            if categorie not in [1, 2, 3]:
                print("  ⚠️  Catégorie invalide, utilisation de 2 par défaut")
                categorie = 2
            
            self.test_scenario(
                name="Test Personnalisé",
                description="Valeurs entrées par l'utilisateur",
                ETo=ETo,
                VPD=VPD,
                soil_moisture=soil_moisture,
                soil_temp=soil_temp,
                categorie=categorie
            )
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Test annulé par l'utilisateur")
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
    
    def test_dataset_sample(self, dataset_path, n_samples=5):
        """Teste sur un échantillon du dataset original"""
        
        print("\n" + "="*70)
        print("📊 TEST SUR ÉCHANTILLON DU DATASET")
        print("="*70)
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset introuvable: {dataset_path}")
            return
        
        # Charger dataset
        df = pd.read_csv(dataset_path)
        
        # Prendre échantillon aléatoire
        sample = df.sample(n=min(n_samples, len(df)))
        
        print(f"\n  📋 {len(sample)} échantillons aléatoires du dataset\n")
        
        for idx, row in sample.iterrows():
            # Valeurs réelles
            real_pump = row['pump_status']
            real_volume = row['irrigation_volume']
            
            # Prédiction
            pred = self.predict(
                row['ETo'],
                row['VPD'],
                row['soil_moisture'],
                row['soil_temp'],
                row['categorie']
            )
            
            # Comparaison
            pump_match = "✅" if pred['pump_status'] == real_pump else "❌"
            volume_error = abs(pred['irrigation_volume'] - real_volume)
            
            print(f"  Ligne {idx}:")
            print(f"    Pompe:  Réel={real_pump} | Prédit={pred['pump_status']} {pump_match}")
            print(f"    Volume: Réel={real_volume:.2f} | Prédit={pred['irrigation_volume']:.2f} | Erreur={volume_error:.2f} mm")
            print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test des modèles ESP32')
    parser.add_argument('--models-dir', default='models_esp32', help='Dossier des modèles')
    parser.add_argument('--mode', choices=['auto', 'custom', 'dataset'], default='auto',
                        help='Mode de test: auto (scénarios), custom (interactif), dataset (échantillon)')
    parser.add_argument('--dataset', default=r'C:\Users\yassi\OneDrive\Bureau\data ++\data\final_dataset.csv',
                        help='Chemin dataset pour mode dataset')
    parser.add_argument('--samples', type=int, default=5, help='Nombre échantillons pour mode dataset')
    
    args = parser.parse_args()
    
    try:
        tester = ModelTester(args.models_dir)
        tester.load_models()
        
        if args.mode == 'auto':
            tester.test_all_scenarios()
        elif args.mode == 'custom':
            tester.test_custom()
        elif args.mode == 'dataset':
            tester.test_dataset_sample(args.dataset, args.samples)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu par l'utilisateur")
        return 1
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())