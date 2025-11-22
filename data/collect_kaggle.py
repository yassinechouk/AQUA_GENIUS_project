#!/usr/bin/env python3
"""
Script : Collection données type Kaggle
Génère des données réalistes de sol et cultures pour la Tunisie
Auteur: Yassinechouk
Date: 2025-11-21 18:19:16 UTC
"""

import pandas as pd
import numpy as np
import argparse
import logging
import sys
import os
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KaggleDataGenerator:
    """Générateur de données type Kaggle pour l'agriculture tunisienne"""
    
    TUNISIA_PARAMS = {
        'latitude': 36.8065,
        'longitude': 10.1815,
        'climate': 'mediterranean',
        'soil_type': 'clay-loam'
    }
    
    CROPS = {
        1: ['olivier', 'amandier', 'figuier', 'vigne'],
        2: ['blé', 'tomate', 'pomme_de_terre', 'oignon'],
        3: ['maïs', 'pastèque', 'melon', 'piment']
    }
    
    def __init__(self):
        pass
    
    def generate_soil_data(self, n_samples: int, start_date: str) -> pd.DataFrame:
        """Génère des données de sol réalistes pour la Tunisie"""
        logger.info(f"🌾 Génération de {n_samples} échantillons de données sol...")
        
        np.random.seed(42)
        
        start = pd.to_datetime(start_date)
        dates = [start + timedelta(days=i) for i in range(n_samples)]
        
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        soil_temp_base = 15 + 15 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
        soil_temp = soil_temp_base + np.random.normal(0, 2, n_samples)
        soil_temp = np.clip(soil_temp, 5, 40)
        
        soil_moisture_base = 50 - (soil_temp - 20) * 1.5
        soil_moisture = soil_moisture_base + np.random.normal(0, 5, n_samples)
        soil_moisture = np.clip(soil_moisture, 15, 70)
        
        soil_ph = np.random.normal(8.0, 0.3, n_samples)
        soil_ph = np.clip(soil_ph, 7.0, 8.8)
        
        soil_ec = np.random.normal(1.5, 0.5, n_samples)
        soil_ec = np.clip(soil_ec, 0.5, 4.0)
        
        nitrogen = np.random.normal(30, 10, n_samples)
        nitrogen = np.clip(nitrogen, 10, 80)
        
        phosphorus = np.random.normal(15, 5, n_samples)
        phosphorus = np.clip(phosphorus, 5, 40)
        
        potassium = np.random.normal(200, 50, n_samples)
        potassium = np.clip(potassium, 100, 400)
        
        df = pd.DataFrame({
            'date': dates,
            'soil_temp': soil_temp,
            'soil_moisture': soil_moisture,
            'soil_ph': soil_ph,
            'soil_ec': soil_ec,
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'soil_type': ['clay-loam'] * n_samples,
        })
        
        logger.info(f"  ✅ Données de sol générées")
        logger.info(f"    • Température sol: {soil_temp.min():.1f}°C - {soil_temp.max():.1f}°C")
        logger.info(f"    • Humidité sol: {soil_moisture.min():.1f}% - {soil_moisture.max():.1f}%")
        logger.info(f"    • pH: {soil_ph.min():.1f} - {soil_ph.max():.1f}")
        
        return df
    
    def generate_crop_data(self, n_samples: int) -> pd.DataFrame:
        """Génère des données de cultures tunisiennes"""
        logger.info(f"🌱 Génération de données de cultures...")
        
        np.random.seed(42)
        
        n_cat1 = int(n_samples * 0.30)
        n_cat2 = int(n_samples * 0.50)
        n_cat3 = n_samples - n_cat1 - n_cat2
        
        categories = []
        crop_names = []
        
        for _ in range(n_cat1):
            categories.append(1)
            crop_names.append(np.random.choice(self.CROPS[1]))
        
        for _ in range(n_cat2):
            categories.append(2)
            crop_names.append(np.random.choice(self.CROPS[2]))
        
        for _ in range(n_cat3):
            categories.append(3)
            crop_names.append(np.random.choice(self.CROPS[3]))
        
        indices = np.random.permutation(n_samples)
        categories = np.array(categories)[indices]
        crop_names = np.array(crop_names)[indices]
        
        kc_map = {1: 0.95, 2: 1.10, 3: 1.20}
        kc_values = [kc_map[cat] for cat in categories]
        
        root_depth_map = {1: 120, 2: 60, 3: 80}
        root_depth = [root_depth_map[cat] + np.random.randint(-10, 10) for cat in categories]
        
        growth_stage = np.random.randint(1, 6, n_samples)
        
        df = pd.DataFrame({
            'categorie': categories,
            'crop_name': crop_names,
            'Kc': kc_values,
            'root_depth_cm': root_depth,
            'growth_stage': growth_stage,
        })
        
        logger.info(f"  ✅ Données de cultures générées")
        for cat in [1, 2, 3]:
            count = (df['categorie'] == cat).sum()
            logger.info(f"    • Catégorie {cat}: {count} ({count/len(df)*100:.1f}%)")
        
        return df
    
    def generate_irrigation_history(self, n_samples: int, soil_moisture: np.ndarray) -> pd.DataFrame:
        """Génère un historique d'irrigation basé sur l'humidité du sol"""
        logger.info(f"💧 Génération de l'historique d'irrigation...")
        
        np.random.seed(42)
        
        days_since_irrigation = []
        last_irrigation_volume = []
        
        for sm in soil_moisture:
            if sm < 30:
                days = np.random.randint(0, 3)
                volume = np.random.uniform(8, 15)
            elif sm < 45:
                days = np.random.randint(2, 7)
                volume = np.random.uniform(5, 10)
            else:
                days = np.random.randint(5, 15)
                volume = np.random.uniform(0, 5)
            
            days_since_irrigation.append(days)
            last_irrigation_volume.append(volume)
        
        df = pd.DataFrame({
            'days_since_irrigation': days_since_irrigation,
            'last_irrigation_volume_mm': last_irrigation_volume,
            'irrigation_method': np.random.choice(
                ['goutte_à_goutte', 'aspersion', 'surface'], 
                n_samples, 
                p=[0.6, 0.3, 0.1]
            ),
        })
        
        logger.info(f"  ✅ Historique d'irrigation généré")
        
        return df
    
    def combine_datasets(self, soil_df: pd.DataFrame, crop_df: pd.DataFrame, 
                        irrigation_df: pd.DataFrame) -> pd.DataFrame:
        """Combine tous les datasets"""
        logger.info(f"🔗 Combinaison des datasets...")
        
        df = pd.concat([soil_df, crop_df, irrigation_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]
        
        logger.info(f"  ✅ Dataset combiné: {len(df)} lignes × {len(df.columns)} colonnes")
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_file: str):
        """Sauvegarde les données"""
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Données sauvegardées: {output_file}")
        
        logger.info(f"\n📊 STATISTIQUES FINALES:")
        logger.info(f"  • Total: {len(df)} lignes")
        logger.info(f"  • Colonnes: {len(df.columns)}")
        logger.info(f"  • Période: {df['date'].min()} → {df['date'].max()}")
        
        if 'categorie' in df.columns:
            logger.info(f"\n  🌾 Catégories:")
            for cat in sorted(df['categorie'].unique()):
                count = (df['categorie'] == cat).sum()
                logger.info(f"    • Catégorie {cat}: {count} ({count/len(df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Génération données type Kaggle")
    parser.add_argument('--samples', type=int, default=1800, help='Nombre d\'échantillons (défaut: 1800)')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Date début (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/kaggle_data.csv', help='Fichier de sortie')
    
    args = parser.parse_args()
    
    try:
        logger.info("="*70)
        logger.info("🌾 GÉNÉRATION DONNÉES TYPE KAGGLE - Agriculture Tunisienne")
        logger.info("="*70)
        logger.info(f"📅 Date: 2025-11-21 18:19:16 UTC")
        logger.info(f"👤 Utilisateur: Yassinechouk")
        logger.info(f"📍 Localisation: Tunisie (36.8065°N, 10.1815°E)")
        logger.info("="*70)
        
        generator = KaggleDataGenerator()
        
        soil_df = generator.generate_soil_data(args.samples, args.start)
        crop_df = generator.generate_crop_data(args.samples)
        irrigation_df = generator.generate_irrigation_history(
            args.samples, 
            soil_df['soil_moisture'].values
        )
        
        final_df = generator.combine_datasets(soil_df, crop_df, irrigation_df)
        generator.save_data(final_df, args.output)
        
        logger.info("\n" + "="*70)
        logger.info("✅ GÉNÉRATION TERMINÉE!")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()