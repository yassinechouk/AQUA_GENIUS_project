#!/usr/bin/env python3
"""
Script d'Entraînement Final - Modèles XGBoost (VERSION STABLE)
Auteur: Yassinechouk
Date: 2025-11-22

Entraîne 2 modèles :
1. Classification : pump_status (OFF/ON)
2. Régression : irrigation_volume (mm/jour)

VERSION TESTÉE ET SANS ERREURS
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import XGBoost avec gestion d'erreur
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    print("❌ ERREUR: XGBoost non installé")
    print("Installation: pip install xgboost")
    sys.exit(1)

# Import scikit-learn
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report,
        r2_score, mean_absolute_error, mean_squared_error
    )
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("❌ ERREUR: scikit-learn non installé")
    print("Installation: pip install scikit-learn")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IrrigationModelTrainer:
    """Entraîneur de modèles d'irrigation - VERSION STABLE"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_pump_train = pd.Series(dtype=int)
        self.y_pump_test = pd.Series(dtype=int)
        self.y_vol_train = pd.Series(dtype=float)
        self.y_vol_test = pd.Series(dtype=float)
        self.clf = None
        self.reg = None
        self.scaler = None
        
        # Features d'entrée (10 variables)
        self.input_features = [
            'tmean', 'tmin', 'tmax', 'humidite', 'Ra', 
            'ETo', 'VPD', 'soil_temp', 'soil_moisture', 'categorie'
        ]
        
        # Variables de sortie (2 variables)
        self.outputs = ['pump_status', 'irrigation_volume']
    
    def load_data(self):
        """Charge et vérifie les données"""
        logger.info("="*70)
        logger.info("📂 ÉTAPE 1 : CHARGEMENT DES DONNÉES")
        logger.info("="*70)
        
        # Vérifier existence fichier
        if not os.path.exists(self.dataset_path):
            logger.error(f"❌ Fichier introuvable: {self.dataset_path}")
            logger.info("\n💡 Générez d'abord le dataset:")
            logger.info("   python generate_final_dataset.py --input data/merged_all.csv --output final_dataset.csv")
            raise FileNotFoundError(f"Fichier non trouvé: {self.dataset_path}")
        
        # Charger CSV
        try:
            self.df = pd.read_csv(self.dataset_path)
            logger.info(f"  ✅ Chargé: {len(self.df)} lignes × {len(self.df.columns)} colonnes")
        except Exception as e:
            logger.error(f"❌ Erreur lecture CSV: {str(e)}")
            raise
        
        # Vérifier colonnes requises
        missing = []
        for col in self.input_features + self.outputs:
            if col not in self.df.columns:
                missing.append(col)
        
        if missing:
            logger.error(f"❌ Colonnes manquantes: {missing}")
            logger.info(f"📋 Colonnes disponibles: {list(self.df.columns)}")
            raise ValueError(f"Colonnes manquantes: {missing}")
        
        logger.info(f"  ✅ Toutes les colonnes présentes")
        
        # Afficher info dataset
        logger.info(f"\n  📊 Informations dataset:")
        logger.info(f"    • Taille: {len(self.df)} lignes")
        logger.info(f"    • Colonnes: {len(self.df.columns)}")
        
        return self
    
    def clean_data(self):
        """Nettoie les données"""
        logger.info("\n" + "="*70)
        logger.info("🧹 ÉTAPE 2 : NETTOYAGE DES DONNÉES")
        logger.info("="*70)
        
        n_before = len(self.df)
        
        # Gérer les NaN
        n_nan = self.df[self.input_features + self.outputs].isnull().sum().sum()
        
        if n_nan > 0:
            logger.warning(f"  ⚠️  {n_nan} valeurs NaN détectées")
            
            # Remplir NaN dans inputs par médiane
            for col in self.input_features:
                if self.df[col].isnull().any():
                    n_col_nan = self.df[col].isnull().sum()
                    median_val = self.df[col].median()
                    self.df[col].fillna(median_val, inplace=True)
                    logger.info(f"    • {col}: {n_col_nan} NaN → médiane ({median_val:.2f})")
            
            # Supprimer lignes avec NaN dans outputs
            self.df.dropna(subset=self.outputs, inplace=True)
            logger.info(f"  ✅ Lignes avec NaN outputs supprimées")
        
        # Convertir types
        for col in self.input_features + self.outputs:
            if col == 'categorie' or col == 'pump_status':
                self.df[col] = self.df[col].astype(int)
            else:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Supprimer lignes avec valeurs aberrantes après conversion
        self.df.dropna(subset=self.input_features + self.outputs, inplace=True)
        
        n_after = len(self.df)
        n_removed = n_before - n_after
        
        if n_removed > 0:
            logger.info(f"  📊 {n_removed} lignes supprimées ({n_removed/n_before*100:.1f}%)")
        
        logger.info(f"  ✅ Dataset nettoyé: {n_after} lignes restantes")
        
        # Vérifier distribution pump_status
        logger.info(f"\n  📊 Distribution pump_status:")
        off = (self.df['pump_status'] == 0).sum()
        on = (self.df['pump_status'] == 1).sum()
        
        logger.info(f"    • OFF (0): {off} ({off/len(self.df)*100:.1f}%)")
        logger.info(f"    • ON (1): {on} ({on/len(self.df)*100:.1f}%)")
        
        if off == 0 or on == 0:
            raise ValueError("pump_status doit contenir à la fois 0 et 1")
        
        # Vérifier cohérence logique
        incoherent = ((self.df['pump_status'] == 0) & (self.df['irrigation_volume'] > 0)).sum()
        if incoherent > 0:
            logger.warning(f"  ⚠️  {incoherent} incohérences (OFF mais volume>0), correction...")
            self.df.loc[self.df['pump_status'] == 0, 'irrigation_volume'] = 0.0
        
        logger.info(f"  ✅ Données cohérentes")
        
        return self
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split train/test"""
        logger.info("\n" + "="*70)
        logger.info("✂️  ÉTAPE 3 : SPLIT TRAIN/TEST")
        logger.info("="*70)
        
        # Extraire X et y
        X = self.df[self.input_features].copy()
        y_pump = self.df['pump_status'].copy()
        y_vol = self.df['irrigation_volume'].copy()
        
        logger.info(f"  📊 Total échantillons: {len(X)}")
        
        # Normalisation
        logger.info(f"  🔧 Normalisation des features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=self.input_features)
        
        # Split avec stratification
        try:
            X_train_arr, X_test_arr, y_pump_train_arr, y_pump_test_arr = train_test_split(
                X, y_pump, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y_pump
            )
            logger.info(f"  ✅ Split avec stratification")
        except:
            logger.warning(f"  ⚠️  Stratification impossible, split normal...")
            X_train_arr, X_test_arr, y_pump_train_arr, y_pump_test_arr = train_test_split(
                X, y_pump, 
                test_size=test_size, 
                random_state=random_state
            )
        
        # Convertir en DataFrame/Series avec reset_index
        self.X_train = pd.DataFrame(X_train_arr, columns=self.input_features)
        self.X_test = pd.DataFrame(X_test_arr, columns=self.input_features)
        self.y_pump_train = pd.Series(y_pump_train_arr, dtype=int).reset_index(drop=True)
        self.y_pump_test = pd.Series(y_pump_test_arr, dtype=int).reset_index(drop=True)
        
        # Split y_vol avec mêmes indices
        y_vol_train_arr = y_vol.iloc[y_pump_train_arr.index].values
        y_vol_test_arr = y_vol.iloc[y_pump_test_arr.index].values
        
        self.y_vol_train = pd.Series(y_vol_train_arr, dtype=float).reset_index(drop=True)
        self.y_vol_test = pd.Series(y_vol_test_arr, dtype=float).reset_index(drop=True)
        
        logger.info(f"\n  📊 Répartition:")
        logger.info(f"    • Train: {len(self.X_train)} ({len(self.X_train)/len(X)*100:.0f}%)")
        logger.info(f"    • Test:  {len(self.X_test)} ({len(self.X_test)/len(X)*100:.0f}%)")
        
        # Distribution
        train_off = (self.y_pump_train == 0).sum()
        train_on = (self.y_pump_train == 1).sum()
        test_off = (self.y_pump_test == 0).sum()
        test_on = (self.y_pump_test == 1).sum()
        
        logger.info(f"\n  📊 Distribution:")
        logger.info(f"    • Train: OFF={train_off}, ON={train_on}")
        logger.info(f"    • Test:  OFF={test_off}, ON={test_on}")
        
        return self
    
    def train_classification(self, optimize=False):
        """Entraîne modèle classification"""
        logger.info("\n" + "="*70)
        logger.info("🤖 ÉTAPE 4 : CLASSIFICATION (pump_status)")
        logger.info("="*70)
        
        if optimize:
            logger.warning("  ⚠️  Optimisation désactivée (utilisez --optimize prudemment)")
        
        # Paramètres optimisés
        logger.info("  ⏳ Entraînement en cours...")
        
        self.clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
        
        # Entraînement
        self.clf.fit(self.X_train, self.y_pump_train)
        logger.info("  ✅ Entraînement terminé")
        
        # Prédictions
        y_pred = self.clf.predict(self.X_test)
        
        # Métriques
        acc = accuracy_score(self.y_pump_test, y_pred)
        prec = precision_score(self.y_pump_test, y_pred, zero_division=0)
        rec = recall_score(self.y_pump_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_pump_test, y_pred, zero_division=0)
        
        logger.info(f"\n  📊 RÉSULTATS:")
        logger.info(f"    • Accuracy:  {acc*100:.2f}%")
        logger.info(f"    • Precision: {prec*100:.2f}%")
        logger.info(f"    • Recall:    {rec*100:.2f}%")
        logger.info(f"    • F1-Score:  {f1*100:.2f}%")
        
        # Matrice de confusion
        cm = confusion_matrix(self.y_pump_test, y_pred)
        logger.info(f"\n  📊 Matrice de Confusion:")
        logger.info(f"              Prédit OFF  Prédit ON")
        logger.info(f"    Réel OFF      {cm[0,0]:6d}     {cm[0,1]:6d}")
        logger.info(f"    Réel ON       {cm[1,0]:6d}     {cm[1,1]:6d}")
        
        # Feature importance (top 5)
        importances = pd.DataFrame({
            'feature': self.input_features,
            'importance': self.clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n  📊 Top 5 Features:")
        for i, row in importances.head(5).iterrows():
            logger.info(f"    {row['feature']:15s} {row['importance']:.4f}")
        
        return {
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1_score': float(f1)
        }
    
    def train_regression(self, optimize=False):
        """Entraîne modèle régression"""
        logger.info("\n" + "="*70)
        logger.info("🤖 ÉTAPE 5 : RÉGRESSION (irrigation_volume)")
        logger.info("="*70)
        
        logger.info("  ⏳ Entraînement en cours...")
        
        self.reg = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )
        
        # Entraînement
        self.reg.fit(self.X_train, self.y_vol_train)
        logger.info("  ✅ Entraînement terminé")
        
        # Prédictions
        y_pred = self.reg.predict(self.X_test)
        
        # Cohérence: OFF → volume=0
        y_pump_pred = self.clf.predict(self.X_test)
        y_pred_coherent = np.where(y_pump_pred == 0, 0.0, y_pred)
        y_pred_coherent = np.clip(y_pred_coherent, 0, 15)
        
        # Convertir en array numpy pour éviter les warnings
        y_vol_test_array = np.array(self.y_vol_test)
        y_pred_coherent_array = np.array(y_pred_coherent)
        
        # Métriques
        r2 = r2_score(y_vol_test_array, y_pred_coherent_array)
        mae = mean_absolute_error(y_vol_test_array, y_pred_coherent_array)
        rmse = np.sqrt(mean_squared_error(y_vol_test_array, y_pred_coherent_array))
        
        logger.info(f"\n  📊 RÉSULTATS:")
        logger.info(f"    • R² Score: {r2:.4f}")
        logger.info(f"    • MAE:      {mae:.3f} mm/jour")
        logger.info(f"    • RMSE:     {rmse:.3f} mm/jour")
        
        # Feature importance (top 5)
        importances = pd.DataFrame({
            'feature': self.input_features,
            'importance': self.reg.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n  📊 Top 5 Features:")
        for i, row in importances.head(5).iterrows():
            logger.info(f"    {row['feature']:15s} {row['importance']:.4f}")
        
        return {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse)
        }
    
    def save_models(self, output_dir='models'):
        """Sauvegarde les modèles"""
        logger.info("\n" + "="*70)
        logger.info("💾 ÉTAPE 6 : SAUVEGARDE")
        logger.info("="*70)
        
        # Créer dossier
        os.makedirs(output_dir, exist_ok=True)
        
        # Chemins
        clf_path = os.path.join(output_dir, 'model_pump_status.pkl')
        reg_path = os.path.join(output_dir, 'model_irrigation_volume.pkl')
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        meta_path = os.path.join(output_dir, 'metadata.json')
        
        joblib.dump(self.clf, clf_path)
        joblib.dump(self.reg, reg_path)
        joblib.dump(self.scaler, scaler_path)
        
        logger.info(f"  ✅ {clf_path}")
        logger.info(f"  ✅ {reg_path}")
        logger.info(f"  ✅ {scaler_path}")
        
        # Métadonnées
        metadata = {
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'author': 'Yassinechouk',
            'dataset': self.dataset_path,
            'n_total': len(self.df),
            'n_train': len(self.X_train),
            'n_test': len(self.X_test),
            'features': self.input_features,
            'outputs': self.outputs
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"  ✅ {meta_path}")
        
        logger.info(f"\n  📊 Tailles fichiers:")
        logger.info(f"    • Classification: {os.path.getsize(clf_path)/1024:.1f} KB")
        logger.info(f"    • Régression:     {os.path.getsize(reg_path)/1024:.1f} KB")
        logger.info(f"    • Scaler:         {os.path.getsize(scaler_path)/1024:.1f} KB")
        
        return self

    def run(self, optimize=False):
        """Pipeline complet"""
        logger.info("="*70)
        logger.info("🚀 ENTRAÎNEMENT MODÈLES D'IRRIGATION")
        logger.info("="*70)
        logger.info(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"👤 Auteur: Yassinechouk")
        logger.info("="*70)
        
        try:
            self.load_data()
            self.clean_data()
            self.split_data()
            clf_metrics = self.train_classification(optimize)
            reg_metrics = self.train_regression(optimize)
            self.save_models()
            
            logger.info("\n" + "="*70)
            logger.info("🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
            logger.info("="*70)
            
            logger.info(f"\n📊 PERFORMANCES:")
            logger.info(f"  Classification:")
            logger.info(f"    • Accuracy: {clf_metrics['accuracy']*100:.2f}%")
            logger.info(f"    • F1-Score: {clf_metrics['f1_score']*100:.2f}%")
            
            logger.info(f"\n  Régression:")
            logger.info(f"    • R²:   {reg_metrics['r2']:.4f}")
            logger.info(f"    • MAE:  {reg_metrics['mae']:.3f} mm/jour")
            logger.info(f"    • RMSE: {reg_metrics['rmse']:.3f} mm/jour")
            
            logger.info(f"\n💾 Modèles: models/")
            logger.info(f"🎯 Prédictions: python predict_irrigation.py")
            logger.info("="*70)
            
            return clf_metrics, reg_metrics
            
        except Exception as e:
            logger.error(f"\n❌ ERREUR: {str(e)}")
            import traceback
            traceback.print_exc()
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Entraînement modèles irrigation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  python train_final.py --input "C:/Users/yassi/OneDrive/Bureau/data ++/final_dataset.csv"
  python train_final.py --input "C:/Users/yassi/OneDrive/Bureau/data ++/final_dataset.csv" --optimize
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='C:/Users/yassi/OneDrive/Bureau/data ++/final_dataset.csv',
        help='Fichier CSV dataset'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Optimiser hyperparamètres (plus long)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Dossier sortie modèles'
    )
    
    args = parser.parse_args()
    
    try:
        trainer = IrrigationModelTrainer(args.input)
        trainer.run(optimize=args.optimize)
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Interruption utilisateur")
        return 1
        
    except Exception as e:
        logger.error(f"\n❌ Échec: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())