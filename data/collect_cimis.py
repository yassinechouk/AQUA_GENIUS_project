#!/usr/bin/env python3
"""
Script de collecte des données CIMIS (California Irrigation Management Information System)
Fournit : ETo, température, humidité, radiation, vitesse du vent
"""

import pandas as pd
import numpy as np
import requests
import argparse
import logging
import sys
import time
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CIMISCollector:
    """Collecteur de données CIMIS"""
    
    BASE_URL = "http://et.water.ca.gov/api/data"
    
    def __init__(self, app_key: str = None):
        """
        Initialise le collecteur CIMIS
        
        Args:
            app_key: Clé API CIMIS (optionnelle pour démo)
        """
        self.app_key = app_key or "demo-key"
    
    def fetch_data(self, station_id: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Récupère les données CIMIS
        
        Args:
            station_id: ID de la station CIMIS
            start_date: Date début (YYYY-MM-DD)
            end_date: Date fin (YYYY-MM-DD)
        
        Returns:
            DataFrame avec les données
        """
        logger.info(f"📡 Requête CIMIS station {station_id}...")
        logger.info(f"  📅 Période: {start_date} → {end_date}")
        
        # Générer des données simulées (car CIMIS nécessite une vraie clé API)
        # Pour production, décommentez la section API ci-dessous
        
        # SIMULATION (pour démonstration)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)
        
        # Simuler des données réalistes pour la Tunisie
        np.random.seed(42)
        
        df = pd.DataFrame({
            'date': dates,
            'station_id': station_id,
            
            # Températures (°C)
            'tmin': 5 + 15 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 2, n),
            'tmax': 15 + 20 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 3, n),
            'tmean': 10 + 17.5 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 2, n),
            
            # Humidité (%)
            'humidite': 60 + 20 * np.sin(2 * np.pi * (np.arange(n) + 180) / 365) + np.random.normal(0, 10, n),
            
            # ETo (mm/jour) - fourni directement par CIMIS
            'ETo': 2 + 4 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 0.5, n),
            
            # Radiation solaire (MJ/m²/jour)
            'solar_radiation': 15 + 20 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 2, n),
            
            # Vitesse du vent (m/s)
            'wind_speed': 2 + np.random.normal(0, 0.5, n),
            
            # Température du sol (°C)
            'soil_temp': 10 + 17 * np.sin(2 * np.pi * np.arange(n) / 365) + np.random.normal(0, 2, n),
            
            # Humidité du sol (%)
            'soil_moisture': 25 + 15 * np.sin(2 * np.pi * (np.arange(n) + 90) / 365) + np.random.normal(0, 5, n),
        })
        
        # Clipper les valeurs
        df['tmin'] = df['tmin'].clip(lower=-5, upper=30)
        df['tmax'] = df['tmax'].clip(lower=10, upper=45)
        df['tmean'] = df['tmean'].clip(lower=0, upper=40)
        df['humidite'] = df['humidite'].clip(lower=20, upper=100)
        df['ETo'] = df['ETo'].clip(lower=0.5, upper=12)
        df['solar_radiation'] = df['solar_radiation'].clip(lower=5, upper=45)
        df['wind_speed'] = df['wind_speed'].clip(lower=0, upper=10)
        df['soil_temp'] = df['soil_temp'].clip(lower=0, upper=40)
        df['soil_moisture'] = df['soil_moisture'].clip(lower=10, upper=50)
        
        logger.info(f"  ✅ {len(df)} jours récupérés (simulé)")
        
        # --- SECTION API RÉELLE (à décommenter pour production) ---
        """
        params = {
            'appKey': self.app_key,
            'targets': station_id,
            'startDate': start_date,
            'endDate': end_date,
            'dataItems': 'day-air-tmp-avg,day-air-tmp-max,day-air-tmp-min,day-rel-hum-avg,day-eto,day-sol-rad-avg,day-wind-spd-avg'
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parser la réponse CIMIS
            records = data.get('Data', {}).get('Providers', [{}])[0].get('Records', [])
            
            df = pd.DataFrame(records)
            # Renommer et traiter les colonnes selon format CIMIS
            
        except Exception as e:
            logger.error(f"❌ Erreur API CIMIS: {e}")
            raise
        """
        
        return df
    
    def calculate_vpd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule le VPD (Vapor Pressure Deficit)"""
        logger.info("  → Calcul de VPD...")
        
        # Pression de vapeur saturante (kPa)
        es = 0.6108 * np.exp((17.27 * df['tmean']) / (df['tmean'] + 237.3))
        
        # Pression de vapeur actuelle (kPa)
        ea = es * (df['humidite'] / 100.0)
        
        # VPD (kPa)
        df['VPD'] = (es - ea).clip(lower=0)
        
        return df
    
    def calculate_ra(self, df: pd.DataFrame, latitude: float) -> pd.DataFrame:
        """Calcule la radiation extraterrestre (Ra)"""
        logger.info("  → Calcul de Ra...")
        
        # Jour julien
        df['doy'] = df['date'].dt.dayofyear
        
        # Déclinaison solaire
        declinaison = 0.409 * np.sin(2 * np.pi * df['doy'] / 365 - 1.39)
        
        # Latitude en radians
        lat_rad = np.radians(latitude)
        
        # Angle horaire coucher de soleil
        ws = np.arccos(-np.tan(lat_rad) * np.tan(declinaison))
        
        # Distance relative Terre-Soleil
        dr = 1 + 0.033 * np.cos(2 * np.pi * df['doy'] / 365)
        
        # Ra (MJ/m²/jour)
        Gsc = 0.0820  # Constante solaire
        df['Ra'] = (24 * 60 / np.pi) * Gsc * dr * (
            ws * np.sin(lat_rad) * np.sin(declinaison) +
            np.cos(lat_rad) * np.cos(declinaison) * np.sin(ws)
        )
        
        df['Ra'] = df['Ra'].clip(lower=0)
        df = df.drop('doy', axis=1)
        
        return df
    
    def process_data(self, df: pd.DataFrame, latitude: float) -> pd.DataFrame:
        """Traite et enrichit les données"""
        logger.info("🔄 Traitement des données CIMIS...")
        
        df = self.calculate_vpd(df)
        df = self.calculate_ra(df, latitude)
        
        logger.info("  ✅ Traitement terminé")
        
        return df
    
    def save_data(self, df: pd.DataFrame, output_file: str):
        """Sauvegarde les données"""
        df.to_csv(output_file, index=False)
        logger.info(f"✅ Données sauvegardées: {output_file}")
        logger.info(f"📊 {len(df)} lignes × {len(df.columns)} colonnes")


def main():
    parser = argparse.ArgumentParser(description="Collecte données CIMIS")
    parser.add_argument('--station', type=int, default=2, help='ID station CIMIS (défaut: 2)')
    parser.add_argument('--start', type=str, default='2020-01-01', help='Date début (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2024-11-21', help='Date fin (YYYY-MM-DD)')
    parser.add_argument('--lat', type=float, default=36.8065, help='Latitude')
    parser.add_argument('--output', type=str, default='data/cimis_data.csv', help='Fichier de sortie')
    parser.add_argument('--api-key', type=str, default=None, help='Clé API CIMIS (optionnel)')
    
    args = parser.parse_args()
    
    try:
        collector = CIMISCollector(app_key=args.api_key)
        
        # Collecter
        df = collector.fetch_data(args.station, args.start, args.end)
        
        # Traiter
        df = collector.process_data(df, args.lat)
        
        # Sauvegarder
        import os
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        collector.save_data(df, args.output)
        
        logger.info("\n✅ Collection CIMIS terminée!")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()