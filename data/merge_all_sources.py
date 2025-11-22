#!/usr/bin/env python3
"""
Script de fusion : NASA POWER + Kaggle + CIMIS
Génère un dataset avec 3 catégories équilibrées
"""

import pandas as pd
import numpy as np
import argparse
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MultiSourceMerger:
    """Fusionneur multi-sources avec catégories"""
    
    def __init__(self):
        pass
    
    def load_data(self, file_path: str, source_name: str) -> pd.DataFrame:
        """Charge un fichier de données"""
        if not file_path or not os.path.exists(file_path):
            logger.info(f"  ℹ️  {source_name} non disponible")
            return None
        
        logger.info(f"📂 Chargement {source_name}: {file_path}")
        df = pd.read_csv(file_path)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        logger.info(f"  ✅ {len(df)} lignes")
        return df
    
    def merge_sources(self, df_nasa, df_kaggle, df_cimis) -> pd.DataFrame:
        """Fusionne les 3 sources"""
        logger.info("🔗 Fusion des sources...")
        
        # Commencer avec NASA (source principale)
        df_merged = df_nasa.copy() if df_nasa is not None else pd.DataFrame()
        
        if df_merged.empty:
            logger.error("❌ Aucune source principale (NASA)")
            sys.exit(1)
        
        # Fusionner Kaggle
        if df_kaggle is not None and len(df_kaggle) > 0:
            logger.info("  → Fusion avec Kaggle...")
            df_merged = pd.merge(df_merged, df_kaggle, on='date', how='outer', suffixes=('', '_kaggle'))
            
            # Combiner colonnes en doublon
            for col in df_merged.columns:
                if col.endswith('_kaggle'):
                    base_col = col.replace('_kaggle', '')
                    if base_col in df_merged.columns:
                        df_merged[base_col] = df_merged[base_col].fillna(df_merged[col])
                    else:
                        df_merged[base_col] = df_merged[col]
                    df_merged = df_merged.drop(col, axis=1)
        
        # Fusionner CIMIS
        if df_cimis is not None and len(df_cimis) > 0:
            logger.info("  → Fusion avec CIMIS...")
            df_merged = pd.merge(df_merged, df_cimis, on='date', how='outer', suffixes=('', '_cimis'))
            
            # Combiner colonnes en doublon
            for col in df_merged.columns:
                if col.endswith('_cimis'):
                    base_col = col.replace('_cimis', '')
                    if base_col in df_merged.columns:
                        df_merged[base_col] = df_merged[base_col].fillna(df_merged[col])
                    else:
                        df_merged[base_col] = df_merged[col]
                    df_merged = df_merged.drop(col, axis=1)
        
        logger.info(f"  ✅ Fusion terminée: {len(df_merged)} lignes")
        
        return df_merged
    
    def assign_categories_balanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assigne les 3 catégories de manière ÉQUILIBRÉE
        Distribution: 30% cat1, 50% cat2, 20% cat3
        """
        logger.info("🌾 Attribution des 3 catégories...")
        
        n = len(df)
        
        # Calculer les nombres par catégorie
        n_cat1 = int(n * 0.30)  # 30%
        n_cat2 = int(n * 0.50)  # 50%
        n_cat3 = n - n_cat1 - n_cat2  # 20%
        
        # Créer les catégories
        categories = np.concatenate([
            np.ones(n_cat1, dtype=int) * 1,
            np.ones(n_cat2, dtype=int) * 2,
            np.ones(n_cat3, dtype=int) * 3
        ])
        
        # Mélanger aléatoirement
        np.random.seed(42)
        np.random.shuffle(categories)
        
        df['categorie'] = categories
        
        # Statistiques
        logger.info(f"  ✅ Distribution des catégories:")
        for cat in [1, 2, 3]:
            count = (df['categorie'] == cat).sum()
            pct = count / len(df) * 100
            logger.info(f"    • Catégorie {cat}: {count} ({pct:.1f}%)")
        
        return df
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Valide et nettoie le dataset"""
        logger.info("✔️  Validation et nettoyage...")
        
        # Colonnes requises
        required = ['date', 'tmean', 'tmin', 'tmax', 'humidite', 'ETo']
        
        missing = [col for col in required if col not in df.columns]
        if missing:
            logger.warning(f"  ⚠️  Colonnes manquantes: {missing}")
        
        # Remplir les NaN
        for col in ['tmin', 'tmax', 'tmean', 'humidite', 'soil_moisture', 'soil_temp']:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Supprimer lignes avec trop de NaN
        df = df.dropna(subset=['date', 'tmean'], how='any')
        
        # Trier
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"  ✅ Dataset nettoyé: {len(df)} lignes")
        
        return df


def main():
    parser = argparse.ArgumentParser(description="Fusion NASA + Kaggle + CIMIS")
    parser.add_argument('--nasa', type=str, required=True, help='Fichier NASA POWER')
    parser.add_argument('--kaggle', type=str, default=None, help='Fichier Kaggle (optionnel)')
    parser.add_argument('--cimis', type=str, default=None, help='Fichier CIMIS (optionnel)')
    parser.add_argument('--output', type=str, default='data/merged_all.csv', help='Fichier de sortie')
    
    args = parser.parse_args()
    
    try:
        merger = MultiSourceMerger()
        
        # Charger les sources
        df_nasa = merger.load_data(args.nasa, "NASA POWER")
        df_kaggle = merger.load_data(args.kaggle, "Kaggle")
        df_cimis = merger.load_data(args.cimis, "CIMIS")
        
        # Fusionner
        df_merged = merger.merge_sources(df_nasa, df_kaggle, df_cimis)
        
        # Assigner les 3 catégories
        df_merged = merger.assign_categories_balanced(df_merged)
        
        # Valider et nettoyer
        df_merged = merger.validate_and_clean(df_merged)
        
        # Sauvegarder
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df_merged.to_csv(args.output, index=False)
        
        logger.info(f"\n✅ DATASET FUSIONNÉ SAUVEGARDÉ: {args.output}")
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 STATISTIQUES FINALES:")
        logger.info(f"  • Total lignes: {len(df_merged)}")
        logger.info(f"  • Total colonnes: {len(df_merged.columns)}")
        logger.info(f"  • Période: {df_merged['date'].min()} → {df_merged['date'].max()}")
        
        logger.info(f"\n🌾 CATÉGORIES:")
        for cat in [1, 2, 3]:
            count = (df_merged['categorie'] == cat).sum()
            pct = count / len(df_merged) * 100
            logger.info(f"  • Catégorie {cat}: {count} ({pct:.1f}%)")
        
        logger.info(f"\n{'='*70}")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()