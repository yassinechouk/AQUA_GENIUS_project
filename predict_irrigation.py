#!/usr/bin/env python3
"""
Script de Prédiction - Système d'Irrigation Intelligente
Auteur: Yassinechouk
Date: 2025-11-22

Utilise les modèles entraînés pour prédire :
1. pump_status (OFF/ON)
2. irrigation_volume (mm/jour)

VERSION SANS WARNINGS PYLANCE
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IrrigationPredictor:
    """Prédicteur pour système d'irrigation intelligente"""
    
    def __init__(self, model_dir: str = 'models') -> None:
        self.model_dir = model_dir
        self.clf: Optional[Any] = None
        self.reg: Optional[Any] = None
        self.scaler: Optional[Any] = None
        self.metadata: Optional[Dict[str, Any]] = None
        self.input_features: List[str] = []
        
        self.load_models()
    
    def load_models(self) -> None:
        """Charge les modèles et métadonnées"""
        logger.info("="*70)
        logger.info("📦 CHARGEMENT DES MODÈLES")
        logger.info("="*70)
        
        # Chemins
        clf_path = os.path.join(self.model_dir, 'model_pump_status.pkl')
        reg_path = os.path.join(self.model_dir, 'model_irrigation_volume.pkl')
        scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
        metadata_path = os.path.join(self.model_dir, 'metadata.json')
        
        # Vérifier existence
        missing_files = []
        for path in [clf_path, reg_path, scaler_path, metadata_path]:
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            logger.error(f"❌ Fichiers manquants: {missing_files}")
            raise FileNotFoundError(
                f"Entraînez d'abord les modèles avec train_irrigation_model.py\n"
                f"Fichiers manquants: {missing_files}"
            )
        
        # Charger
        try:
            self.clf = joblib.load(clf_path)
            self.reg = joblib.load(reg_path)
            self.scaler = joblib.load(scaler_path)
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            # Extraire features avec vérification (support de 'features' OU 'input_features')
            if self.metadata:
                if 'input_features' in self.metadata:
                    self.input_features = self.metadata['input_features']
                elif 'features' in self.metadata:
                    self.input_features = self.metadata['features']
                else:
                    # Fallback si metadata mal formée
                    self.input_features = [
                        'tmean', 'tmin', 'tmax', 'humidite', 'Ra', 
                        'ETo', 'VPD', 'soil_temp', 'soil_moisture', 'categorie'
                    ]
                    logger.warning("⚠️  Métadonnées incomplètes, utilisation des features par défaut")
            else:
                # Si metadata est None
                self.input_features = [
                    'tmean', 'tmin', 'tmax', 'humidite', 'Ra', 
                    'ETo', 'VPD', 'soil_temp', 'soil_moisture', 'categorie'
                ]
                logger.warning("⚠️  Métadonnées non disponibles, utilisation des features par défaut")
            
            logger.info(f"  ✅ Modèle classification chargé")
            logger.info(f"  ✅ Modèle régression chargé")
            logger.info(f"  ✅ Scaler chargé")
            logger.info(f"  ✅ Métadonnées chargées")
            logger.info(f"  ✅ Features: {len(self.input_features)} variables")
            
            if self.metadata:
                logger.info(f"\n  📅 Date entraînement: {self.metadata.get('date', 'N/A')}")
                logger.info(f"  📊 Dataset: {self.metadata.get('n_total', 'N/A')} échantillons")
            
        except KeyError as e:
            logger.error(f"❌ Clé manquante dans metadata: {e}")
            logger.info("💡 Utilisation des features par défaut...")
            self.input_features = [
                'tmean', 'tmin', 'tmax', 'humidite', 'Ra', 
                'ETo', 'VPD', 'soil_temp', 'soil_moisture', 'categorie'
            ]
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèles: {e}")
            raise
    
    def predict_single(self, data: Dict[str, float]) -> Dict[str, Any]:
        """
        Prédiction pour un seul échantillon
        
        Args:
            data: dict avec les 10 features requises
            
        Returns:
            dict avec pump_status et irrigation_volume
        """
        # Vérifier que les modèles sont chargés
        if self.clf is None or self.reg is None or self.scaler is None:
            raise RuntimeError("Les modèles ne sont pas chargés correctement")
        
        # Vérifier features
        missing = [f for f in self.input_features if f not in data]
        if missing:
            raise ValueError(f"Features manquantes: {missing}")
        
        # Créer DataFrame
        df = pd.DataFrame([data], columns=self.input_features)
        
        # Normaliser
        X_scaled = self.scaler.transform(df)
        
        # Prédire
        pump_status = int(self.clf.predict(X_scaled)[0])
        pump_proba = self.clf.predict_proba(X_scaled)[0]
        
        irrigation_volume = float(self.reg.predict(X_scaled)[0])
        
        # Cohérence: si OFF, volume = 0
        if pump_status == 0:
            irrigation_volume = 0.0
        
        # Clipper volume
        irrigation_volume = np.clip(irrigation_volume, 0, 15)
        
        return {
            'pump_status': pump_status,
            'pump_status_label': 'ON' if pump_status == 1 else 'OFF',
            'pump_proba_off': float(pump_proba[0]),
            'pump_proba_on': float(pump_proba[1]),
            'irrigation_volume': round(irrigation_volume, 2)
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prédiction pour un batch (DataFrame)
        
        Args:
            df: DataFrame avec colonnes = input_features
            
        Returns:
            DataFrame avec colonnes ajoutées: pump_status, irrigation_volume
        """
        # Vérifier que les modèles sont chargés
        if self.clf is None or self.reg is None or self.scaler is None:
            raise RuntimeError("Les modèles ne sont pas chargés correctement")
        
        logger.info(f"\n🔮 Prédiction sur {len(df)} échantillons...")
        
        # Vérifier colonnes
        missing = [f for f in self.input_features if f not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes: {missing}")
        
        # Extraire features
        X = df[self.input_features].copy()
        
        # Normaliser
        X_scaled = self.scaler.transform(X)
        
        # Prédire
        pump_status = self.clf.predict(X_scaled)
        pump_proba = self.clf.predict_proba(X_scaled)
        irrigation_volume = self.reg.predict(X_scaled)
        
        # Cohérence
        irrigation_volume = np.where(pump_status == 0, 0.0, irrigation_volume)
        irrigation_volume = np.clip(irrigation_volume, 0, 15)
        
        # Ajouter au DataFrame
        df_result = df.copy()
        df_result['pump_status'] = pump_status
        df_result['pump_proba_off'] = pump_proba[:, 0]
        df_result['pump_proba_on'] = pump_proba[:, 1]
        df_result['irrigation_volume'] = irrigation_volume
        
        # Statistiques
        off_count = (pump_status == 0).sum()
        on_count = (pump_status == 1).sum()
        
        logger.info(f"  ✅ Prédictions terminées")
        logger.info(f"  📊 Pump OFF: {off_count} ({off_count/len(df)*100:.1f}%)")
        logger.info(f"  📊 Pump ON: {on_count} ({on_count/len(df)*100:.1f}%)")
        logger.info(f"  📊 Volume moyen: {irrigation_volume.mean():.2f} mm/jour")
        
        return df_result
    
    def predict_from_csv(self, input_csv: str, output_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Prédiction depuis fichier CSV
        
        Args:
            input_csv: fichier d'entrée
            output_csv: fichier de sortie (optionnel)
        
        Returns:
            DataFrame avec prédictions
        """
        logger.info("="*70)
        logger.info("🔮 PRÉDICTION DEPUIS CSV")
        logger.info("="*70)
        
        # Charger
        if not os.path.exists(input_csv):
            raise FileNotFoundError(f"Fichier introuvable: {input_csv}")
        
        df = pd.read_csv(input_csv)
        logger.info(f"  📂 {len(df)} lignes chargées depuis {input_csv}")
        
        # Prédire
        df_result = self.predict_batch(df)
        
        # Sauvegarder
        if output_csv:
            df_result.to_csv(output_csv, index=False)
            logger.info(f"\n  💾 Résultats sauvegardés: {output_csv}")
        
        return df_result


def main() -> int:
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prédiction irrigation intelligente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:

1. Prédiction interactive (1 échantillon):
   python predict_irrigation.py --interactive

2. Prédiction depuis CSV:
   python predict_irrigation.py --input new_data.csv --output predictions.csv

3. Avec répertoire de modèles personnalisé:
   python predict_irrigation.py --models my_models/ --input data.csv
        """
    )
    
    parser.add_argument(
        '--models',
        type=str,
        default='models',
        help='Répertoire des modèles (défaut: models/)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Fichier CSV d\'entrée pour prédiction batch'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Fichier CSV de sortie (défaut: predictions_YYYYMMDD_HHMMSS.csv)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Mode interactif pour une prédiction unique'
    )
    
    args = parser.parse_args()
    
    try:
        # Charger modèles
        predictor = IrrigationPredictor(model_dir=args.models)
        
        if args.interactive:
            # Mode interactif
            logger.info("\n" + "="*70)
            logger.info("💬 MODE INTERACTIF")
            logger.info("="*70)
            logger.info("Entrez les 10 features:")
            
            data: Dict[str, float] = {}
            for feature in predictor.input_features:
                while True:
                    try:
                        val = input(f"  {feature}: ")
                        if feature == 'categorie':
                            data[feature] = float(int(val))
                        else:
                            data[feature] = float(val)
                        break
                    except ValueError:
                        print(f"    ⚠️  Valeur invalide, réessayez")
            
            # Prédire
            result = predictor.predict_single(data)
            
            logger.info("\n" + "="*70)
            logger.info("🎯 RÉSULTAT DE LA PRÉDICTION")
            logger.info("="*70)
            logger.info(f"  🚰 Statut Pompe: {result['pump_status_label']}")
            logger.info(f"  📊 Probabilités: OFF={result['pump_proba_off']*100:.1f}%, ON={result['pump_proba_on']*100:.1f}%")
            logger.info(f"  💧 Volume Irrigation: {result['irrigation_volume']:.2f} mm/jour")
            logger.info("="*70)
            
        elif args.input:
            # Mode batch
            output = args.output
            if not output:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output = f'predictions_{timestamp}.csv'
            
            predictor.predict_from_csv(args.input, output)
            
            logger.info("\n✅ Prédictions terminées avec succès!")
            
        else:
            logger.error("❌ Spécifiez --input ou --interactive")
            parser.print_help()
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interruption utilisateur")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())