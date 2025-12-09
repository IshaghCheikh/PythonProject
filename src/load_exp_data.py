import requests
import pandas as pd
import os
import time

class MultiZoneMeteoLoader:
    """
    Charge les variables météorologiques explicatives pour plusieurs zones
    afin de construire un proxy national pour la prévision PV.
    """
    
    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        # Coordonnées des centroïdes de production solaire en France
        # Ces points couvrent >60% de la capacité installée française.
        self.zones = {
            'Bordeaux': {'lat': 44.83, 'lon': -0.57}, # Représente Nouvelle-Aquitaine
            'Marseille': {'lat': 43.29, 'lon': 5.37}, # Représente PACA
            'Lyon':      {'lat': 45.76, 'lon': 4.83}, # Représente Rhône-Alpes
            'Nantes':    {'lat': 47.21, 'lon': -1.55},# Représente l'Ouest
            'Paris':     {'lat': 48.85, 'lon': 2.35}  # Représente le Nord/Centre
        }
        
        # Variables physiques demandées
        self.params_template = {
            "hourly": "shortwave_radiation,direct_normal_irradiance,diffuse_radiation,temperature_2m,wind_speed_10m,cloud_cover",
            "timezone": "UTC"
        }

    def fetch_zone_data(self, zone_name: str, coords: dict, start_date: str, end_date: str) -> pd.DataFrame:
        """Récupère les données pour une zone spécifique."""
        print(f"📡 Récupération météo pour {zone_name}...")
        
        params = self.params_template.copy()
        params['latitude'] = coords['lat']
        params['longitude'] = coords['lon']
        params['start_date'] = start_date
        params['end_date'] = end_date
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df.set_index('time', inplace=True)
            
            # Renommage scientifique avec suffixe de zone
            # Ex: shortwave_radiation -> GHI_Bordeaux
            mapping = {
                'shortwave_radiation': f'GHI_{zone_name}',
                'direct_normal_irradiance': f'DNI_{zone_name}',
                'diffuse_radiation': f'DHI_{zone_name}',
                'temperature_2m': f'T_{zone_name}',
                'wind_speed_10m': f'WS_{zone_name}',
                'cloud_cover': f'Cloud_{zone_name}'
            }
            df.rename(columns=mapping, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Erreur pour {zone_name}: {e}")
            return pd.DataFrame()

    def build_national_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Agrège les données de toutes les zones dans un seul DataFrame large.
        """
        national_df = pd.DataFrame()
        
        for zone, coords in self.zones.items():
            df_zone = self.fetch_zone_data(zone, coords, start_date, end_date)
            
            if national_df.empty:
                national_df = df_zone
            else:
                # Fusionner sur l'index temporel (Join)
                national_df = national_df.join(df_zone, how='outer')
            
            # Pause pour respecter les limites API
            time.sleep(0.5)
            
        return national_df

# --- MAIN ---
if __name__ == "__main__":
    # Définition de la période (doit correspondre à vos données de production)
    START_DATE = "2018-01-01"
    END_DATE = "2023-12-31"
    FOLDER = "/content/PythonProject/data"
    
    loader = MultiZoneMeteoLoader()
    
    print(f"--- Démarrage de l'extraction multizone ({START_DATE} à {END_DATE}) ---")
    df_meteo = loader.build_national_dataset(START_DATE, END_DATE)
    
    if not df_meteo.empty:
        # Création du dossier
        os.makedirs(FOLDER, exist_ok=True)
        
        # Sauvegarde
        filename = f"meteo_nationale_{START_DATE}_{END_DATE}.csv"
        path = os.path.join(FOLDER, filename)
        df_meteo.to_csv(path)
        
        print(f"\n✅ Succès ! Matrice de variables explicatives générée.")
        print(f"Dimensions : {df_meteo.shape}")
        print(f"Colonnes ({len(df_meteo.columns)}) : {df_meteo.columns.tolist()[:5]} ...")
        print(f"Sauvegardé dans : {path}")
    else:
        print("❌ Échec de la récupération des données.")