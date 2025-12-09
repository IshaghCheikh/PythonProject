import requests
import pandas as pd
import io
import time

class FrenchSolarLoader:
    """
    Récupère les données de production solaire en France via l'API ODRÉ (RTE).
    """
    
    def __init__(self):
        # API Base URL pour Open Data Réseaux Énergies
        self.api_url = "https://odre.opendatasoft.com/api/v2/catalog/datasets/eco2mix-national-cons-def/exports/csv"
        # Note: 'eco2mix-national-tr' contient le temps réel ET l'historique récent.
        # Pour l'historique consolidé ancien, on utiliserait 'eco2mix-national-cons-def'.

    def fetch_national_solar(self, year: int) -> pd.DataFrame:
        """
        Télécharge la production solaire nationale au pas 30 min pour une année donnée.
        
        Args:
            year (int): Année à récupérer (ex: 2023).
            
        Returns:
            pd.DataFrame: Index datetime UTC, colonne 'Y_production' (MW).
        """
        print(f"🇫🇷 [RTE] Téléchargement des données France pour {year}...")
        
        # Filtres ODRÉ (SQL-like)
        # On filtre sur l'année pour ne pas surcharger la requête
        where_clause = f"date_heure >= '{year}-01-01' AND date_heure <= '{year}-12-31'"
        
        params = {
            'where': where_clause,
            'limit': -1,  # -1 demande toutes les lignes (attention à la limite de taille)
            'select': 'date_heure, solaire', # On ne prend que la date et le solaire
            'timezone': 'UTC', # Important pour l'alignement scientifique
            'delimiter': ';'
        }
        
        try:
            response = requests.get(self.api_url, params=params)
            response.raise_for_status()
            
            # Lecture du CSV reçu
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=';')
            
            # Nettoyage
            if 'date_heure' in df.columns:
                df['time'] = pd.to_datetime(df['date_heure'], utc=True)
                df.set_index('time', inplace=True)
                df.drop(columns=['date_heure'], inplace=True)
            
            # Renommage
            df.rename(columns={'solaire': 'Y_production'}, inplace=True)
            
            # Tri temporel (l'API ne garantit pas l'ordre)
            df.sort_index(inplace=True)
            
            # Suppression des lignes sans production (la nuit) ou manquantes
            # Note scientifique : La nuit, la production est 0, c'est une info valide !
            # On ne supprime que les NaN (données manquantes)
            df['Y_production'] = df['Y_production'].fillna(0)
            
            print(f"✅ Année {year} récupérée : {df.shape[0]} points.")
            return df
            
        except Exception as e:
            print(f"❌ Erreur pour {year} : {e}")
            return pd.DataFrame()

    def fetch_multiple_years(self, start_year: int, end_year: int) -> pd.DataFrame:
        all_dfs = []
        for y in range(start_year, end_year + 1):
            df = self.fetch_national_solar(y)
            all_dfs.append(df)
            time.sleep(1) # Pause respectueuse pour l'API
            
        if not all_dfs:
            return pd.DataFrame()
            
        # Concaténation de tout l'historique
        full_df = pd.concat(all_dfs)
        full_df.sort_index(inplace=True)
        
        # Gestion des doublons éventuels
        full_df = full_df[~full_df.index.duplicated(keep='first')]
        
        return full_df

# --- UTILISATION ---
if __name__ == "__main__":
    loader = FrenchSolarLoader()
    
    # Récupération de 2022 à 2023 (Données d'entraînement classiques)
    df_france = loader.fetch_multiple_years(2018, 2023)
    
    print("\n--- RÉSUMÉ DU DATASET FRANCE ---")
    print(df_france.info())
    print(df_france.head())

    # Sauvegarde
    df_france.to_csv("/content/PythonProject/data/production_solaire_france_2018_2023.csv")
    print("💾 Fichier sauvegardé : production_solaire_france_2018_2023.csv")
    
   