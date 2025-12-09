import pandas as pd
import numpy as np
import os

def process_pipeline_30min():
    # --- CONFIGURATION ---
    # Chemins vers vos fichiers bruts téléchargés précédemment
    INPUT_FILE_PROD = "data/production_solaire_france_2018_2023.csv" # Source 15 min ou 30 min
    INPUT_FILE_METEO = "data/meteo_nationale_2018-01-01_2023-12-31.csv"          # Source 1h
    OUTPUT_FILE = "data/processed/dataset_complet_30min.csv"
    
    print("--- DÉMARRAGE DU PIPELINE DE FUSION (Cible : 30 min) ---")

    # 1. Chargement des données brutes
    print("1. Chargement des CSV...")
    try:
        df_y = pd.read_csv(INPUT_FILE_PROD, index_col=0, parse_dates=True)
        df_x = pd.read_csv(INPUT_FILE_METEO, index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        print(f"❌ ERREUR : Fichier manquant. {e}")
        return

    # Standardisation Timezone (UTC pour éviter les bugs d'été/hiver)
    if df_y.index.tz is None: df_y.index = df_y.index.tz_localize('UTC')
    if df_x.index.tz is None: df_x.index = df_x.index.tz_localize('UTC')
    
    # Conversion explicite en UTC
    df_y.index = df_y.index.tz_convert('UTC')
    df_x.index = df_x.index.tz_convert('UTC')

    print(f"   Shape Prod brute  : {df_y.shape} (Freq détectée: {pd.infer_freq(df_y.index)})")
    print(f"   Shape Météo brute : {df_x.shape} (Freq détectée: {pd.infer_freq(df_x.index)})")

    # 2. Traitement de la Production (Y) : Agrégation vers 30 min
    # Si vos données sont en 15 min, on fait la MOYENNE pour passer à 30 min.
    # Si elles sont déjà en 30 min, cette opération ne change rien (idempotente).
    print("\n2. Normalisation Production -> 30 min (Mean Aggregation)...")
    df_y_30 = df_y.resample('30T').mean()
    
    # 3. Traitement de la Météo (X) : Interpolation vers 30 min
    print("3. Normalisation Météo -> 30 min (Linear Interpolation)...")
    # On force la grille 30 min
    df_x_30 = df_x.resample('30T').asfreq()
    # On comble les trous par linéarité
    df_x_30 = df_x_30.interpolate(method='linear')

    # 4. Fusion (Inner Join)
    print("\n4. Fusion des datasets (Alignement temporel)...")
    df_final = df_y_30.join(df_x_30, how='inner')

    # 5. Nettoyage
    # Suppression des NaNs (causés par des décalages au début/fin de fichier)
    len_before = len(df_final)
    df_final.dropna(inplace=True)
    len_after = len(df_final)
    print(f"   Lignes supprimées (NaN) : {len_before - len_after}")

    # 6. Feature Engineering (Indispensable pour le ML)
    # Encodage cyclique de l'heure et du mois
    df_final['hour_sin'] = np.sin(2 * np.pi * df_final.index.hour / 24)
    df_final['hour_cos'] = np.cos(2 * np.pi * df_final.index.hour / 24)
    df_final['month_sin'] = np.sin(2 * np.pi * df_final.index.month / 12)
    df_final['month_cos'] = np.cos(2 * np.pi * df_final.index.month / 12)

    # 7. Sauvegarde et Audit
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_csv(OUTPUT_FILE)
    
    print("\n" + "="*40)
    print(f"✅ SUCCÈS : Dataset 30min généré.")
    print(f"📁 Chemin  : {OUTPUT_FILE}")
    print(f"📊 Dimensions finales : {df_final.shape}")
    print(f"📅 Période : {df_final.index.min()} à {df_final.index.max()}")
    print("="*40)
    
    # Vérification dimensionnelle théorique pour 6 ans (2018-2023)
    # 2191 jours * 48 points = 105 168 lignes
    if abs(len(df_final) - 105168) < 2000:
        print("INFO : La taille correspond aux attentes théoriques (~105k lignes).")
    else:
        print("ATTENTION : Écart significatif de taille. Vérifiez vos fichiers sources.")

if __name__ == "__main__":
    process_pipeline_30min()