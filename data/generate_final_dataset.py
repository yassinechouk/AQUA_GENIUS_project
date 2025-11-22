#!/usr/bin/env python3
"""
Script ULTIME : Génération du Dataset Final
Auteur: Yassinechouk
Date: 2025-11-21 18:55:53 UTC

Ce script génère le dataset final parfait avec :
✅ 10 INPUTS + 2 OUTPUTS
✅ Unités correctes et conversions
✅ Cohérences logiques et physiques
✅ Distribution équilibrée (50% OFF / 50% ON)
✅ 3 catégories (30% / 50% / 20%)
✅ Correction automatique soil_moisture
✅ Aucune incohérence
"""

import pandas as pd
import numpy as np
import argparse
import logging
import sys
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FinalDatasetGenerator:
    """Générateur du dataset final parfait"""
    
    # ========================================================================
    # DÉFINITION DES VARIABLES
    # ========================================================================
    
    INPUTS = {
        'tmean': {'unit': '°C', 'min': 5, 'max': 45, 'description': 'Température moyenne'},
        'tmin': {'unit': '°C', 'min': 0, 'max': 40, 'description': 'Température minimale'},
        'tmax': {'unit': '°C', 'min': 10, 'max': 50, 'description': 'Température maximale'},
        'humidite': {'unit': '%', 'min': 20, 'max': 95, 'description': 'Humidité relative'},
        'Ra': {'unit': 'MJ/m²/j', 'min': 10, 'max': 45, 'description': 'Radiation extraterrestre'},
        'ETo': {'unit': 'mm/j', 'min': 0.5, 'max': 12.0, 'description': 'Évapotranspiration de référence'},
        'VPD': {'unit': 'kPa', 'min': 0.1, 'max': 5.0, 'description': 'Déficit de pression de vapeur'},
        'soil_temp': {'unit': '°C', 'min': 5, 'max': 40, 'description': 'Température du sol'},
        'soil_moisture': {'unit': '%', 'min': 10, 'max': 70, 'description': 'Humidité du sol'},
        'categorie': {'unit': '', 'min': 1, 'max': 3, 'description': 'Catégorie de culture'}
    }
    
    OUTPUTS = {
        'pump_status': {'unit': '', 'values': [0, 1], 'description': 'Statut pompe (0=OFF, 1=ON)'},
        'irrigation_volume': {'unit': 'mm/j', 'min': 0.0, 'max': 15.0, 'description': 'Volume d\'irrigation'}
    }
    
    # Coefficients Kc par catégorie
    KC_VALUES = {
        1: 0.95,  # Faibles besoins (olivier, amandier)
        2: 1.10,  # Besoins modérés (blé, tomate)
        3: 1.20   # Besoins élevés (maïs, pastèque)
    }
    
    # Distribution des catégories
    CATEGORY_DISTRIBUTION = {
        1: 0.30,  # 30%
        2: 0.50,  # 50%
        3: 0.20   # 20%
    }
    
    def __init__(self):
        pass
    
    # ========================================================================
    # ÉTAPE 1 : CHARGEMENT ET FUSION
    # ========================================================================
    
    def load_merged_data(self, input_file: str) -> pd.DataFrame:
        """Charge le dataset fusionné"""
        logger.info("="*70)
        logger.info("📂 ÉTAPE 1 : CHARGEMENT DES DONNÉES")
        logger.info("="*70)
        
        if not os.path.exists(input_file):
            logger.error(f"❌ Fichier introuvable: {input_file}")
            logger.info("\n💡 Générez d'abord les données:")
            logger.info("   python merge_all_sources.py --nasa data/nasa_power.csv --kaggle data/kaggle_tunisia.csv --output data/merged_all.csv")
            sys.exit(1)
        
        df = pd.read_csv(input_file)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        logger.info(f"  ✅ {len(df)} lignes × {len(df.columns)} colonnes")
        logger.info(f"  📅 Période: {df['date'].min()} → {df['date'].max()}")
        
        return df
    
    # ========================================================================
    # ÉTAPE 2 : SÉLECTION ET CONVERSION DES INPUTS
    # ========================================================================
    
    def select_and_convert_inputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sélectionne et convertit les 10 inputs avec unités correctes"""
        logger.info("\n" + "="*70)
        logger.info("🔧 ÉTAPE 2 : SÉLECTION ET CONVERSION DES INPUTS")
        logger.info("="*70)
        
        df_clean = pd.DataFrame()
        df_clean['date'] = df['date']
        
        for var_name, var_info in self.INPUTS.items():
            logger.info(f"\n  • {var_name} ({var_info['description']})")
            logger.info(f"    Unité: {var_info['unit']}, Plage: [{var_info['min']}, {var_info['max']}]")
            
            if var_name in df.columns:
                values = df[var_name].copy()
                
                # CONVERSIONS SPÉCIFIQUES
                
                # 1. soil_moisture : convertir 0-1 → 0-100%
                if var_name == 'soil_moisture' and values.max() < 10:
                    logger.warning(f"    ⚠️  Conversion × 100 (format décimal détecté)")
                    values = values * 100
                
                # 2. Température : vérifier cohérence tmin < tmean < tmax
                if var_name == 'tmean':
                    if 'tmin' in df.columns and 'tmax' in df.columns:
                        # Forcer cohérence
                        tmin = df['tmin']
                        tmax = df['tmax']
                        values = np.clip(values, tmin, tmax)
                
                # 3. Clipper dans la plage valide
                values = np.clip(values, var_info['min'], var_info['max'])
                
                # 4. CORRECTION SPÉCIALE pour soil_moisture
                if var_name == 'soil_moisture':
                    # Vérifier si les valeurs sont quasi-constantes
                    if values.std() < 1.0 or pd.Series(values).nunique() < 20:
                        logger.warning(f"    ⚠️  soil_moisture quasi-constant (std={values.std():.2f}, unique={pd.Series(values).nunique()})")
                        logger.warning(f"    → Régénération intelligente...")
                        
                        # Régénérer avec logique cohérente
                        base_moisture = 40  # Moyenne de 40%
                        
                        # Variation basée sur ETo (inversement proportionnel)
                        if 'ETo' in df.columns:
                            eto_effect = -1.5 * (df['ETo'] - df['ETo'].mean())
                        else:
                            eto_effect = 0
                        
                        # Variation saisonnière
                        if 'date' in df_clean.columns:
                            day_of_year = df_clean['date'].dt.dayofyear
                            seasonal_effect = 15 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
                        else:
                            seasonal_effect = 0
                        
                        # Bruit aléatoire
                        np.random.seed(42)
                        random_noise = np.random.normal(0, 5, len(values))
                        
                        # Nouvelle soil_moisture
                        values = base_moisture + eto_effect + seasonal_effect + random_noise
                        values = np.clip(values, var_info['min'], var_info['max'])
                        
                        logger.info(f"    ✅ Régénéré: min={values.min():.1f}, max={values.max():.1f}, mean={values.mean():.1f}, std={values.std():.2f}, unique={values.nunique()}")
                
                # 5. Interpoler les NaN
                if pd.isna(values).sum() > 0:
                    n_nan = pd.isna(values).sum()
                    logger.warning(f"    ⚠️  {n_nan} NaN, interpolation...")
                    values = pd.Series(values).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
                
                df_clean[var_name] = values
                
                logger.info(f"    ✅ min={values.min():.2f}, max={values.max():.2f}, mean={values.mean():.2f}")
                
            elif var_name == 'categorie':
                # Générer les catégories
                logger.warning(f"    ⚠️  Catégorie manquante, génération...")
                df_clean = self.generate_categories(df_clean)
                logger.info(f"    ✅ Catégories générées (30% / 50% / 20%)")
            else:
                logger.error(f"    ❌ Variable manquante: {var_name}")
                sys.exit(1)
        
        return df_clean
    
    def generate_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère les 3 catégories avec distribution 30/50/20"""
        n = len(df)
        n_cat1 = int(n * 0.30)
        n_cat2 = int(n * 0.50)
        n_cat3 = n - n_cat1 - n_cat2
        
        categories = np.concatenate([
            np.ones(n_cat1, dtype=int) * 1,
            np.ones(n_cat2, dtype=int) * 2,
            np.ones(n_cat3, dtype=int) * 3
        ])
        
        np.random.seed(42)
        np.random.shuffle(categories)
        df['categorie'] = categories
        
        return df
    
    # ========================================================================
    # ÉTAPE 3 : VÉRIFICATION DES COHÉRENCES PHYSIQUES
    # ========================================================================
    
    def verify_physical_coherence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vérifie et corrige les incohérences physiques"""
        logger.info("\n" + "="*70)
        logger.info("🔬 ÉTAPE 3 : VÉRIFICATION DES COHÉRENCES PHYSIQUES")
        logger.info("="*70)
        
        n_corrections = 0
        
        # 1. Cohérence températures : tmin ≤ tmean ≤ tmax
        logger.info("\n  • Vérification tmin ≤ tmean ≤ tmax...")
        incoherent = (df['tmin'] > df['tmean']) | (df['tmean'] > df['tmax'])
        n_incoherent = incoherent.sum()
        
        if n_incoherent > 0:
            logger.warning(f"    ⚠️  {n_incoherent} incohérences détectées, correction...")
            
            # Correction : recalculer tmean
            df.loc[incoherent, 'tmean'] = (df.loc[incoherent, 'tmin'] + df.loc[incoherent, 'tmax']) / 2
            n_corrections += n_incoherent
            
            logger.info(f"    ✅ {n_incoherent} corrections appliquées")
        else:
            logger.info(f"    ✅ Aucune incohérence")
        
        # 2. Cohérence ETo vs température
        logger.info("\n  • Vérification ETo cohérent avec température...")
        # ETo doit augmenter avec la température
        eto_expected = 0.0023 * (df['tmean'] + 17.8) * np.sqrt((df['tmax'] - df['tmin']).clip(lower=0)) * df['Ra']
        eto_expected = eto_expected.clip(lower=0.5, upper=12.0)
        
        # Remplacer les valeurs trop éloignées
        eto_diff = np.abs(df['ETo'] - eto_expected)
        large_diff = eto_diff > 3.0
        n_large_diff = large_diff.sum()
        
        if n_large_diff > 0:
            logger.warning(f"    ⚠️  {n_large_diff} valeurs ETo aberrantes, recalcul...")
            df.loc[large_diff, 'ETo'] = eto_expected[large_diff]
            n_corrections += n_large_diff
            logger.info(f"    ✅ {n_large_diff} corrections appliquées")
        else:
            logger.info(f"    ✅ ETo cohérent")
        
        # 3. Cohérence VPD vs humidité
        logger.info("\n  • Vérification VPD cohérent avec humidité...")
        # VPD doit diminuer quand humidité augmente
        es = 0.6108 * np.exp((17.27 * df['tmean']) / (df['tmean'] + 237.3))
        ea = es * (df['humidite'] / 100.0)
        vpd_expected = (es - ea).clip(lower=0.1, upper=5.0)
        
        vpd_diff = np.abs(df['VPD'] - vpd_expected)
        large_diff = vpd_diff > 1.5
        n_large_diff = large_diff.sum()
        
        if n_large_diff > 0:
            logger.warning(f"    ⚠️  {n_large_diff} valeurs VPD aberrantes, recalcul...")
            df.loc[large_diff, 'VPD'] = vpd_expected[large_diff]
            n_corrections += n_large_diff
            logger.info(f"    ✅ {n_large_diff} corrections appliquées")
        else:
            logger.info(f"    ✅ VPD cohérent")
        
        # 4. Cohérence soil_temp vs tmean
        logger.info("\n  • Vérification température sol cohérente...")
        # soil_temp ~ tmean (± quelques degrés)
        soil_temp_expected = df['tmean'] + np.random.normal(0, 2, len(df))
        soil_temp_expected = soil_temp_expected.clip(lower=5, upper=40)
        
        soil_diff = np.abs(df['soil_temp'] - df['tmean'])
        large_diff = soil_diff > 10
        n_large_diff = large_diff.sum()
        
        if n_large_diff > 0:
            logger.warning(f"    ⚠️  {n_large_diff} valeurs soil_temp aberrantes, correction...")
            df.loc[large_diff, 'soil_temp'] = soil_temp_expected[large_diff]
            n_corrections += n_large_diff
            logger.info(f"    ✅ {n_large_diff} corrections appliquées")
        else:
            logger.info(f"    ✅ Température sol cohérente")
        
        logger.info(f"\n  📊 Total corrections physiques: {n_corrections}")
        
        return df
    
    # ========================================================================
    # ÉTAPE 4 : GÉNÉRATION DES OUTPUTS
    # ========================================================================
    
    def generate_outputs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Génère pump_status et irrigation_volume avec logique intelligente"""
        logger.info("\n" + "="*70)
        logger.info("💧 ÉTAPE 4 : GÉNÉRATION DES OUTPUTS")
        logger.info("="*70)
        
        # 1. Calculer le score de besoin d'irrigation
        logger.info("\n  • Calcul du score de besoin d'irrigation...")
        
        # Normaliser les variables
        soil_moisture_norm = (df['soil_moisture'] - df['soil_moisture'].min()) / (df['soil_moisture'].max() - df['soil_moisture'].min())
        eto_norm = (df['ETo'] - df['ETo'].min()) / (df['ETo'].max() - df['ETo'].min())
        temp_norm = (df['tmean'] - df['tmean'].min()) / (df['tmean'].max() - df['tmean'].min())
        vpd_norm = (df['VPD'] - df['VPD'].min()) / (df['VPD'].max() - df['VPD'].min())
        
        # Score multi-critères
        irrigation_need_score = (
            (1 - soil_moisture_norm) * 0.40 +  # 40% : sol sec → besoin élevé
            eto_norm * 0.30 +                   # 30% : ETo élevé → besoin élevé
            temp_norm * 0.15 +                  # 15% : température élevée
            vpd_norm * 0.15                     # 15% : VPD élevé → stress hydrique
        )
        
        # 2. Décision pump_status (seuil médian pour 50/50)
        logger.info("\n  • Attribution pump_status (objectif: 50% OFF / 50% ON)...")
        
        threshold = irrigation_need_score.median()
        df['pump_status'] = (irrigation_need_score > threshold).astype(int)
        
        off_count = (df['pump_status'] == 0).sum()
        on_count = (df['pump_status'] == 1).sum()
        
        logger.info(f"    ✅ Pump OFF: {off_count} ({off_count/len(df)*100:.1f}%)")
        logger.info(f"    ✅ Pump ON: {on_count} ({on_count/len(df)*100:.1f}%)")
        
        # 3. Calculer irrigation_volume
        logger.info("\n  • Calcul irrigation_volume...")
        
        # Besoin en eau = ETo × Kc
        df['Kc'] = df['categorie'].map(self.KC_VALUES)
        water_need = (df['ETo'] * df['Kc']).clip(lower=0, upper=15.0)
        
        # Volume = besoin si ON, sinon 0
        df['irrigation_volume'] = np.where(df['pump_status'] == 1, water_need, 0.0)
        
        # Supprimer Kc temporaire
        df = df.drop('Kc', axis=1)
        
        vol_mean_off = df[df['pump_status'] == 0]['irrigation_volume'].mean()
        vol_mean_on = df[df['pump_status'] == 1]['irrigation_volume'].mean()
        
        logger.info(f"    ✅ Volume moyen OFF: {vol_mean_off:.2f} mm/j")
        logger.info(f"    ✅ Volume moyen ON: {vol_mean_on:.2f} mm/j")
        
        return df
    
    # ========================================================================
    # ÉTAPE 5 : VÉRIFICATION DES COHÉRENCES LOGIQUES
    # ========================================================================
    
    def verify_logical_coherence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Vérifie et corrige les incohérences logiques"""
        logger.info("\n" + "="*70)
        logger.info("🔍 ÉTAPE 5 : VÉRIFICATION DES COHÉRENCES LOGIQUES")
        logger.info("="*70)
        
        # 1. Cohérence : pump_status=0 → irrigation_volume=0
        logger.info("\n  • Vérification : OFF → volume=0...")
        
        incoherent = (df['pump_status'] == 0) & (df['irrigation_volume'] > 0)
        n_incoherent = incoherent.sum()
        
        if n_incoherent > 0:
            logger.warning(f"    ⚠️  {n_incoherent} incohérences détectées, correction...")
            df.loc[incoherent, 'irrigation_volume'] = 0.0
            logger.info(f"    ✅ {n_incoherent} corrections appliquées")
        else:
            logger.info(f"    ✅ Aucune incohérence")
        
        # 2. Cohérence : pump_status=1 → irrigation_volume>0
        logger.info("\n  • Vérification : ON → volume>0...")
        
        incoherent = (df['pump_status'] == 1) & (df['irrigation_volume'] == 0)
        n_incoherent = incoherent.sum()
        
        if n_incoherent > 0:
            logger.warning(f"    ⚠️  {n_incoherent} incohérences détectées, correction...")
            # Calculer un volume minimum basé sur ETo
            df.loc[incoherent, 'irrigation_volume'] = (df.loc[incoherent, 'ETo'] * 0.8).clip(lower=2.0, upper=8.0)
            logger.info(f"    ✅ {n_incoherent} corrections appliquées")
        else:
            logger.info(f"    ✅ Cohérent")
        
        # 3. Vérifier distribution par catégorie
        logger.info("\n  • Vérification distribution par catégorie...")
        
        for cat in [1, 2, 3]:
            df_cat = df[df['categorie'] == cat]
            off_cat = (df_cat['pump_status'] == 0).sum()
            on_cat = (df_cat['pump_status'] == 1).sum()
            
            logger.info(f"    • Catégorie {cat}: OFF={off_cat}, ON={on_cat}")
            
            if off_cat == 0 or on_cat == 0:
                logger.warning(f"      ⚠️  Catégorie {cat} déséquilibrée!")
        
        return df
    
    # ========================================================================
    # ÉTAPE 6 : VALIDATION FINALE
    # ========================================================================
    
    def final_validation(self, df: pd.DataFrame) -> bool:
        """Validation finale complète"""
        logger.info("\n" + "="*70)
        logger.info("✅ ÉTAPE 6 : VALIDATION FINALE")
        logger.info("="*70)
        
        all_valid = True
        
        # 1. Vérifier les 10 inputs
        logger.info("\n  📥 Vérification des 10 INPUTS:")
        for i, (var_name, var_info) in enumerate(self.INPUTS.items(), 1):
            if var_name in df.columns:
                values = df[var_name]
                min_val = values.min()
                max_val = values.max()
                in_range = (min_val >= var_info['min']) and (max_val <= var_info['max'])
                status = "✅" if in_range else "❌"
                
                logger.info(f"    {status} {i:2d}. {var_name:20s} [{min_val:7.2f}, {max_val:7.2f}] {var_info['unit']}")
                
                if not in_range:
                    all_valid = False
            else:
                logger.error(f"    ❌ {var_name} MANQUANT")
                all_valid = False
        
        # 2. Vérifier les 2 outputs
        logger.info("\n  📤 Vérification des 2 OUTPUTS:")
        
        # pump_status
        if 'pump_status' in df.columns:
            unique_vals = sorted(df['pump_status'].unique())
            has_both = (0 in unique_vals) and (1 in unique_vals)
            status = "✅" if has_both else "❌"
            
            off = (df['pump_status'] == 0).sum()
            on = (df['pump_status'] == 1).sum()
            
            logger.info(f"    {status} 1. pump_status: OFF={off}, ON={on}")
            
            if not has_both:
                all_valid = False
        else:
            logger.error(f"    ❌ pump_status MANQUANT")
            all_valid = False
        
        # irrigation_volume
        if 'irrigation_volume' in df.columns:
            min_vol = df['irrigation_volume'].min()
            max_vol = df['irrigation_volume'].max()
            in_range = (min_vol >= 0) and (max_vol <= 15.0)
            status = "✅" if in_range else "❌"
            
            logger.info(f"    {status} 2. irrigation_volume: [{min_vol:.2f}, {max_vol:.2f}] mm/j")
            
            if not in_range:
                all_valid = False
        else:
            logger.error(f"    ❌ irrigation_volume MANQUANT")
            all_valid = False
        
        # 3. Vérifier cohérences
        logger.info("\n  🔍 Vérification cohérences:")
        
        # OFF → volume=0
        incoherent_off = ((df['pump_status'] == 0) & (df['irrigation_volume'] > 0)).sum()
        status_off = "✅" if incoherent_off == 0 else "❌"
        logger.info(f"    {status_off} OFF → volume=0: {incoherent_off} incohérences")
        if incoherent_off > 0:
            all_valid = False
        
        # ON → volume>0
        incoherent_on = ((df['pump_status'] == 1) & (df['irrigation_volume'] == 0)).sum()
        status_on = "✅" if incoherent_on == 0 else "⚠️ "
        logger.info(f"    {status_on} ON → volume>0: {incoherent_on} exceptions")
        
        # 4. Distribution
        logger.info("\n  📊 Distribution finale:")
        logger.info(f"    • Total lignes: {len(df)}")
        logger.info(f"    • Période: {df['date'].min()} → {df['date'].max()}")
        
        for cat in [1, 2, 3]:
            count = (df['categorie'] == cat).sum()
            pct = count / len(df) * 100
            logger.info(f"    • Catégorie {cat}: {count} ({pct:.1f}%)")
        
        return all_valid
    
    # ========================================================================
    # ÉTAPE 7 : SAUVEGARDE
    # ========================================================================
    
    def save_final_dataset(self, df: pd.DataFrame, output_file: str):
        """Sauvegarde le dataset final"""
        logger.info("\n" + "="*70)
        logger.info("💾 ÉTAPE 7 : SAUVEGARDE")
        logger.info("="*70)
        
        # Sélectionner colonnes finales (10 inputs + 2 outputs + date)
        final_columns = ['date'] + list(self.INPUTS.keys()) + list(self.OUTPUTS.keys())
        df_final = df[final_columns].copy()
        
        # Sauvegarder
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        df_final.to_csv(output_file, index=False)
        
        logger.info(f"\n✅ Dataset final sauvegardé: {output_file}")
        logger.info(f"📊 {len(df_final)} lignes × {len(df_final.columns)} colonnes")
        
        # Afficher échantillon
        logger.info(f"\n📋 ÉCHANTILLON (5 premières lignes):")
        print(df_final.head().to_string())
    
    # ========================================================================
    # PIPELINE COMPLET
    # ========================================================================
    
    def generate(self, input_file: str, output_file: str):
        """Pipeline complet de génération"""
        
        logger.info("="*70)
        logger.info("🚀 GÉNÉRATION DU DATASET FINAL")
        logger.info("="*70)
        logger.info(f"📅 Date: 2025-11-21 18:55:53 UTC")
        logger.info(f"👤 Utilisateur: Yassinechouk")
        logger.info("="*70)
        
        # ÉTAPE 1 : Chargement
        df = self.load_merged_data(input_file)
        
        # ÉTAPE 2 : Sélection et conversion inputs
        df = self.select_and_convert_inputs(df)
        
        # ÉTAPE 3 : Vérification cohérences physiques
        df = self.verify_physical_coherence(df)
        
        # ÉTAPE 4 : Génération outputs
        df = self.generate_outputs(df)
        
        # ÉTAPE 5 : Vérification cohérences logiques
        df = self.verify_logical_coherence(df)
        
        # ÉTAPE 6 : Validation finale
        is_valid = self.final_validation(df)
        
        if not is_valid:
            logger.error("\n❌ VALIDATION ÉCHOUÉE - Corrections nécessaires")
            sys.exit(1)
        
        # ÉTAPE 7 : Sauvegarde
        self.save_final_dataset(df, output_file)
        
        logger.info("\n" + "="*70)
        logger.info("🎉 GÉNÉRATION TERMINÉE AVEC SUCCÈS!")
        logger.info("="*70)
        logger.info(f"\n🚀 Prochaine étape:")
        logger.info(f"   python train_final.py --input {output_file} --optimize")
        logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(description="Génération du dataset final parfait")
    parser.add_argument('--input', type=str, required=True, help='Fichier merged (data/merged_all.csv)')
    parser.add_argument('--output', type=str, default='tunisia_irrigation_final_corrected.csv', help='Fichier de sortie')
    
    args = parser.parse_args()
    
    try:
        generator = FinalDatasetGenerator()
        generator.generate(args.input, args.output)
        
    except Exception as e:
        logger.error(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()