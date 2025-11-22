#!/usr/bin/env python3
"""
Script 1/4 : Collecte de données NASA POWER
Génère: tmean, tmin, tmax, humidite, Ra, ETo, VPD, soil_temp, soil_moisture
"""

import requests
import pandas as pd
import numpy as np
import argparse
import logging
from datetime import datetime
import sys
import os
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NASAPowerCollector:
    """Collecteur NASA POWER"""
    
    BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
    PARAMETERS = ["T2M", "T2M_MIN", "T2M_MAX", "RH2M", "TS", "GWETROOT"]
    
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
    
    def fetch_data(self, start_date: str, end_date: str, max_retries: int = 3) -> pd.DataFrame:
        """Récupère les données depuis NASA POWER API"""
        params = {
            "parameters": ",".join(self.PARAMETERS),
            "community": "AG",
            "longitude": self.longitude,
            "latitude": self.latitude,
            "start": start_date,
            "end": end_date,
            "format": "JSON"
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"📡 Requête NASA POWER (tentative {attempt + 1}/{max_retries})...")
                response = requests.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if "properties" not in data or "parameter" not in data["properties"]:
                    raise ValueError("Format de réponse invalide")
                
                parameters_data = data["properties"]["parameter"]
                df = pd.DataFrame(parameters_data)
                df.index = pd.to_datetime(df.index, format='%Y%m%d')
                df.index.name = 'date'
                df = df.reset_index()
                
                logger.info(f"✅ {len(df)} jours récupérés")
                return df
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"❌ Erreur (tentative {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    def calculate_ra(self, latitude: float, doy: int) -> float:
        """Calcule Ra (Radiation Extraterrestre) - FAO-56"""
        lat_rad = np.radians(latitude)
        delta = 0.409 * np.sin((2 * np.pi / 365) * doy - 1.39)
        dr = 1 + 0.033 * np.cos((2 * np.pi / 365) * doy)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        Gsc = 0.0820
        Ra = (24 * 60 / np.pi) * Gsc * dr * (
            ws * np.sin(lat_rad) * np.sin(delta) + 
            np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )
        return max(0, Ra)
    
    def calculate_eto(self, tmin: float, tmax: float, tmean: float, Ra: float) -> float:
        """Calcule ETo - Méthode Hargreaves"""
        eto = 0.0023 * (tmean + 17.8) * np.sqrt(max(0, tmax - tmin)) * Ra
        return max(0, eto)
    
    def calculate_vpd(self, temperature: float, relative_humidity: float) -> float:
        """Calcule VPD (Déficit Pression Vapeur)"""
        es = 0.6108 * np.exp((17.27 * temperature) / (temperature + 237.3))
        ea = es * (relative_humidity / 100.0)
        return max(0, es - ea)
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Traite les données brutes"""
        logger.info("🔄 Traitement des données...")
        
        df['day_of_year'] = pd.to_datetime(df['date']).dt.dayofyear
        
        # Calculer Ra
        logger.info("  → Calcul de Ra...")
        df['Ra'] = df['day_of_year'].apply(lambda doy: self.calculate_ra(self.latitude, doy))
        
        # Calculer ETo
        logger.info("  → Calcul de ETo...")
        df['ETo'] = df.apply(
            lambda row: self.calculate_eto(row['T2M_MIN'], row['T2M_MAX'], row['T2M'], row['Ra']),
            axis=1
        )
        
        # Calculer VPD
        logger.info("  → Calcul de VPD...")
        df['VPD'] = df.apply(
            lambda row: self.calculate_vpd(row['T2M'], row['RH2M']),
            axis=1
        )
        
        # Renommer colonnes
        df = df.rename(columns={
            'T2M': 'tmean',
            'T2M_MIN': 'tmin',
            'T2M_MAX': 'tmax',
            'RH2M': 'humidite',
            'TS': 'soil_temp',
            'GWETROOT': 'soil_moisture'
        })
        
        # Colonnes finales
        final_columns = ['date', 'tmean', 'tmin', 'tmax', 'humidite', 'Ra', 'ETo', 'VPD', 'soil_temp', 'soil_moisture']
        df = df[final_columns]
        df = df.ffill().bfill()
        
        logger.info("✅ Traitement terminé")
        return df


def main():
    parser = argparse.ArgumentParser(description="Script 1/4 - Collecte NASA POWER")
    parser.add_argument('--lat', type=float, required=True, help='Latitude')
    parser.add_argument('--lon', type=float, required=True, help='Longitude')
    parser.add_argument('--start', type=str, required=True, help='Date début (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='Date fin (YYYY-MM-DD)')
    parser.add_argument('--output', type=str, default='data/nasa_power.csv', help='Fichier sortie')
    
    args = parser.parse_args()
    
    try:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').strftime('%Y%m%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d').strftime('%Y%m%d')
        
        collector = NASAPowerCollector(args.lat, args.lon)
        df = collector.fetch_data(start_date, end_date)
        df_processed = collector.process_data(df)
        
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df_processed.to_csv(args.output, index=False)
        
        logger.info(f"✅ Données sauvegardées: {args.output}")
        logger.info(f"📊 {len(df_processed)} lignes × {len(df_processed.columns)} colonnes")
        
    except Exception as e:
        logger.error(f"❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()