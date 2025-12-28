import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def prod_ete_hiver(start_summer , end_summer, start_winter, end_winter, df , ylabel):
    
    # Création de la figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharey=True)

    # Zoom sur une semaine d'ÉTÉ (Juin 2022)
    
    df[start_summer:end_summer][ylabel].plot(ax=axes[0], color='#FF8C00', lw=2)
    axes[0].set_title(f"Profil de Production - ÉTÉ ({start_summer} - {end_summer})", fontweight='bold')
    axes[0].set_ylabel("Production (MW)")

    # Zoom sur une semaine d'HIVER (Décembre 2022)
  
    df[start_winter:end_winter][ylabel].plot(ax=axes[1], color='#1E90FF', lw=2)
    axes[1].set_title(f"Profil de Production - HIVER ({start_winter} - {end_winter})", fontweight='bold')
    axes[1].set_ylabel("Production (MW)")
    plt.tight_layout()
    plt.show()
    #
    
def var_boxplot(df):
        # Cellule : Analyse de la Variance (Boxplots)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. Variabilité par HEURE (Cycle Diurne)
    sns.boxplot(data=df, x=df.index.hour, y='Y_production', ax=axes[0], palette="Oranges")
    axes[0].set_title("Distribution de la Production par Heure de la Journée", fontweight='bold')
    axes[0].set_xlabel("Heure (UTC)")
    axes[0].set_ylabel("Production (MW)")

    # 2. Variabilité par MOIS (Saisonnalité)
    sns.boxplot(data=df, x=df.index.month, y='Y_production', ax=axes[1], palette="Blues")
    axes[1].set_title("Distribution de la Production par Mois", fontweight='bold')
    axes[1].set_xlabel("Mois")
    axes[1].set_ylabel("Production (MW)")

    plt.tight_layout()
    plt.show()



def carpet_plot(df):
    # Cellule : Carpet Plot
    # Préparation des données matricielles
    df['date'] = df.index.date
    df['hour_float'] = df.index.hour + df.index.minute / 60

    # Pivot : Lignes = Jours, Colonnes = Heures
    pivot_table = df.pivot_table(values='Y_production', index='date', columns='hour_float')

    plt.figure(figsize=(12, 8))
    # Utilisation de 'jet' ou 'inferno' pour bien voir les intensités
    sns.heatmap(pivot_table, cmap='jet', cbar_kws={'label': 'Production (MW)'})

    plt.title("Carpet Plot : Production Solaire (Jours vs Heures)", fontweight='bold')
    plt.xlabel("Heure de la journée")
    plt.ylabel("Date")
    plt.show()

    # Nettoyage des colonnes temporaires
    df.drop(columns=['date', 'hour_float'], inplace=True)    



def target_analysis(df):
        # Préparation des données agrégées
    df['year'] = df.index.year
    df['month'] = df.index.month

    # --- 1. Calcul de l'Énergie Totale (Intégration) ---
    # Attention aux unités : On a des MW toutes les 30 min.
    # Énergie (MWh) = Puissance (MW) * Temps (0.5 h)
    annual_energy = df.groupby('year')['Y_production'].sum() * 0.5 / 1000 # Conversion en GWh

    # --- 2. Comparaison des Profils Moyens Mensuels ---
    monthly_profile = df.groupby(['year', 'month'])['Y_production'].mean().unstack(level=0)

    # --- VISUALISATION ---
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Graphique A : Tendance de l'Énergie Produite (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = sns.color_palette("viridis", len(annual_energy))
    annual_energy.plot(kind='bar', ax=ax1, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title("A. Production Totale Annuelle (GWh)", fontweight='bold')
    ax1.set_ylabel("Énergie (GWh)")
    ax1.set_xlabel("Année")
    ax1.grid(axis='y', alpha=0.3)
    # Ajout des valeurs sur les barres
    for i, v in enumerate(annual_energy):
        ax1.text(i, v + 100, f"{int(v)}", ha='center', fontweight='bold')

    # Graphique B : Comparaison des Saisons (Line Plot)
    ax2 = fig.add_subplot(gs[0, 1])
    monthly_profile.plot(ax=ax2, marker='o', linewidth=2, cmap='viridis')
    ax2.set_title("B. Profil Moyen Mensuel par Année", fontweight='bold')
    ax2.set_ylabel("Puissance Moyenne (MW)")
    ax2.set_xlabel("Mois")
    ax2.legend(title='Année', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax2.grid(True, alpha=0.3)

    # Graphique C : Distribution de la Puissance (Violin Plot)
    # Permet de voir l'augmentation de la capacité installée (les pics montent)
    ax3 = fig.add_subplot(gs[1, :]) # Prend toute la largeur du bas
    sns.violinplot(data=df, x='year', y='Y_production', ax=ax3, palette="viridis", hue='year', legend=False)
    ax3.set_title("C. Distribution de la Puissance Instantanée par Année", fontweight='bold')
    ax3.set_ylabel("Production (MW)")
    ax3.set_xlabel("Année")
    ax3.grid(axis='y', alpha=0.3)

    plt.show()

    # Nettoyage des colonnes temporaires
    df.drop(columns=['year', 'month'], inplace=True)


import requests


def fetch_capacity_dynamic_final():
    print("📡 Connexion API RTE (Récupération des données)...")
    
    # 1. Endpoint V2.1
    dataset_id = "parc-national-annuel-prod-eolien-solaire"
    url = f"https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset_id}/records"
    
    # On demande les 50 dernières années
    params = {'limit': 50, 'order_by': 'annee DESC'}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        records = data.get('results', [])
        
        if not records:
            raise ValueError("L'API a répondu mais la liste est vide.")
            
        df_api = pd.DataFrame(records)
        
        # 2. Identification de la colonne Solaire
        # On cherche 'parc_installe_solaire' (votre retour d'erreur précédent)
        possible_cols = ['parc_installe_solaire', 'parc_solaire_mw', 'puissance_solaire']
        
        target_col = None
        for col in possible_cols:
            if col in df_api.columns:
                target_col = col
                break
        
        if target_col is None:
            # Fallback : on prend la première colonne qui contient 'solaire'
            cols_solaire = [c for c in df_api.columns if 'solaire' in c]
            if not cols_solaire:
                raise KeyError(f"Aucune colonne solaire trouvée. Colonnes : {df_api.columns.tolist()}")
            target_col = cols_solaire[0]
            
        print(f"✅ Colonne utilisée : '{target_col}'")
            
        # 3. Nettoyage
        df_clean = df_api[['annee', target_col]].dropna().copy()
        df_clean = df_clean.sort_values('annee')
        
        # 4. Logique Temporelle (Année N -> 31 Décembre N)
        df_clean['date'] = pd.to_datetime(df_clean['annee'].astype(str) + '-12-31').dt.tz_localize('UTC')
        
        ts_capacity = df_clean.set_index('date')[target_col]
        
        # CORRECTION DU BUG ICI : On utilise iloc[-1] pour afficher la dernière valeur
        last_val = ts_capacity.iloc[-1]
        last_date = ts_capacity.index[-1].year
        print(f"✅ SUCCÈS : Données récupérées (Fin {last_date} : {last_val} MW)")
        
        return ts_capacity

    except Exception as e:
        raise RuntimeError(f"❌ Erreur API : {e}")


from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries, title="Test de Stationnarité"):
    """
    Effectue le test de Dickey-Fuller Augmenté (ADF) et plot les statistiques roulantes.
    """
    # 1. Calcul des statistiques roulantes (fenêtre de 1 an = 365*48 points)
    # On prend une grande fenêtre pour lisser la saisonnalité et voir la TENDANCE
    window_size = 365 * 48 
    rolmean = timeseries.rolling(window=window_size).mean()
    rolstd = timeseries.rolling(window=window_size).std()

    # 2. Plot visuel
    plt.figure(figsize=(14, 5))
    plt.plot(timeseries, color='blue', label='Original (Light)', alpha=0.3)
    plt.plot(rolmean, color='red', label='Moyenne Mobile (1 an)')
    plt.plot(rolstd, color='black', label='Écart-type Mobile')
    plt.legend(loc='best')
    plt.title(title, fontweight='bold')
    plt.show()

    # 3. Test de Dickey-Fuller (ADF)
    print(f'--- Résultats du Test ADF pour : {title} ---')
    dftest = adfuller(timeseries.dropna(), autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
        
    print(dfoutput)
    
    # Interprétation automatique pour la thèse
    if dfoutput['p-value'] < 0.05:
        print("✅ CONCLUSION : La série est STATIONNAIRE (H0 rejetée).")
    else:
        print("❌ CONCLUSION : La série est NON-STATIONNAIRE (Unit Root présent).")
    print("-" * 50)   



def plot_correlation_matrix(df, target='Y_load_factor'):
    # 1. Sélection des colonnes numériques pertinentes
    # On exclut les colonnes 'textes' ou les dates
    cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ['Y_production', 'Installed_Capacity']]
    
    # Calcul de la matrice
    corr_matrix = df[cols].corr(method='pearson')
    
    # 2. Visualisation
    plt.figure(figsize=(12, 10))
    
    # Masque pour cacher la partie supérieure (redondante)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, fmt=".2f", cmap='coolwarm', 
                vmin=-1, vmax=1, center=0, linewidths=0.5)
    
    plt.title(f"Matrice de Corrélation de Pearson (Target: {target})", fontweight='bold')
    plt.show()
    
    # 3. Zoom sur la Target
    print(f"--- Top Corrélations avec {target} ---")
    print(corr_matrix[target].sort_values(ascending=False))



def plot_radiation_dependency(df, rad_col, target='Y_load_factor'):
    """
    Visualise la densité de la relation Input (Radiation) -> Output (Prod)
    rad_col : Nom de la colonne de rayonnement (ex: 'ssrd', 'GHI', 'rsds')
    """
    plt.figure(figsize=(10, 6))
    
    # Hexbin pour voir la densité des points
    plt.hexbin(df[rad_col], df[target], gridsize=50, cmap='inferno', mincnt=1)
    
    plt.colorbar(label='Nombre d\'observations')
    plt.xlabel(f"Rayonnement : {rad_col}")
    plt.ylabel("Facteur de Charge (0-1)")
    plt.title(f"Fonction de Transfert : {rad_col} -> PV Output", fontweight='bold')
    plt.grid(True, alpha=0.2)
    plt.show()



def target_analysis(df , target):
    # Préparation des données agrégées
    df['year'] = df.index.year
    df['month'] = df.index.month

    # --- 1. Calcul de l'Énergie Totale (Intégration) ---
    # Attention aux unités : On a des MW toutes les 30 min.
    # Énergie (MWh) = Puissance (MW) * Temps (0.5 h)
    annual_energy = df.groupby('year')[target].sum() * 0.5 / 1000 # Conversion en GWh

    # --- 2. Comparaison des Profils Moyens Mensuels ---
    monthly_profile = df.groupby(['year', 'month'])[target].mean().unstack(level=0)

    # --- VISUALISATION ---
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # Graphique A : Tendance de l'Énergie Produite (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = sns.color_palette("viridis", len(annual_energy))
    annual_energy.plot(kind='bar', ax=ax1, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_title("A. Production Totale Annuelle (GWh)", fontweight='bold')
    ax1.set_ylabel("Énergie (GWh)")
    ax1.set_xlabel("Année")
    ax1.grid(axis='y', alpha=0.3)
    # Ajout des valeurs sur les barres
    for i, v in enumerate(annual_energy):
        ax1.text(i, v + 100, f"{int(v)}", ha='center', fontweight='bold')

    # Graphique B : Comparaison des Saisons (Line Plot)
    ax2 = fig.add_subplot(gs[0, 1])
    monthly_profile.plot(ax=ax2, marker='o', linewidth=2, cmap='viridis')
    ax2.set_title("B. Profil Moyen Mensuel par Année", fontweight='bold')
    ax2.set_ylabel("Puissance Moyenne (MW)")
    ax2.set_xlabel("Mois")
    ax2.legend(title='Année', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
    ax2.grid(True, alpha=0.3)

    # Graphique C : Distribution de la Puissance (Violin Plot)
    # Permet de voir l'augmentation de la capacité installée (les pics montent)
    ax3 = fig.add_subplot(gs[1, :]) # Prend toute la largeur du bas
    sns.violinplot(data=df, x='year', y='Y_production', ax=ax3, palette="viridis", hue='year', legend=False)
    ax3.set_title("C. Distribution de la Puissance Instantanée par Année", fontweight='bold')
    ax3.set_ylabel("Production (MW)")
    ax3.set_xlabel("Année")
    ax3.grid(axis='y', alpha=0.3)

    plt.show()

    # Nettoyage des colonnes temporaires
    df.drop(columns=['year', 'month'], inplace=True)    



def plot_temperature_effect(df, temp_col, rad_col, target='Y_load_factor'):
        """
        Montre l'effet de la température pour un niveau de rayonnement élevé et fixe.
        """
        # On isole les moments de "Grand Soleil" (Haut du panier)
        # Ex: Rayonnement > 90% du max
        high_rad_threshold = df[rad_col].quantile(0.90)
        subset = df[df[rad_col] > high_rad_threshold].copy()
        
        plt.figure(figsize=(10, 6))
        
        sns.regplot(data=subset, x=temp_col, y=target, scatter_kws={'alpha':0.3, 'color':'orange'}, line_kws={'color':'red'})
        
        plt.title(f"Effet de la Température à Rayonnement Constant (> {int(high_rad_threshold)})", fontweight='bold')
        plt.xlabel(f"Température ({temp_col})")
        plt.ylabel("Facteur de Charge")
        plt.grid(True)
        plt.show()
        
        # Calcul de la pente
        corr = subset[[temp_col, target]].corr().iloc[0, 1]
        print(f"📉 Corrélation à haute irradiance : {corr:.3f}")
        if corr < 0:
            print("✅ Hypothèse Physique Validée : La chaleur baisse le rendement.")
        else:
            print("⚠️ Pas d'effet thermique visible (ou données biaisées).")

from statsmodels.stats.outliers_influence import variance_inflation_factor
def plot_vif(x_train):
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = x_train.columns
    vif_data["VIF"] = [variance_inflation_factor(x_train.values, i) for i in range(x_train.shape[1])]

    # Sort by VIF value
    vif_data = vif_data.sort_values('VIF', ascending=False)

    print("=== Variance Inflation Factor (VIF) ===")
    print(vif_data)
    print("\n💡 Interprétation :")
    print("   VIF < 5  : Pas de multicolinéarité")
    print("   VIF 5-10 : Multicolinéarité modérée")
    print("   VIF > 10 : Multicolinéarité FORTE (à corriger)")

    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.barh(vif_data["Feature"], vif_data["VIF"], color=['red' if v > 10 else 'orange' if v > 5 else 'green' for v in vif_data["VIF"]])
    plt.axvline(x=5, color='orange', linestyle='--', label='Seuil Modéré (5)')
    plt.axvline(x=10, color='red', linestyle='--', label='Seuil Critique (10)')
    plt.xlabel('VIF Score')
    plt.title('Test de Multicolinéarité (VIF)', fontweight='bold')
    plt.legend()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

from adjustText import adjust_text  
def plot_correlation_circle_readable(pca, feature_names, pc_x=0, pc_y=1):
    """
    Affiche un cercle des corrélations avec des labels qui ne se chevauchent pas
    grâce à la librairie adjustText.
    """
    # Extraction des loadings
    loadings = pca.components_.T

    fig, ax = plt.subplots(figsize=(10, 10)) # Un peu plus grand pour l'espace

    # 1. Dessiner le cercle unitaire et les axes
    circle = plt.Circle((0, 0), 1, color='grey', fill=False, linestyle='--', alpha=0.5)
    ax.add_artist(circle)
    plt.axhline(0, color='grey', linewidth=1, linestyle='-')
    plt.axvline(0, color='grey', linewidth=1, linestyle='-')

    # Liste pour stocker les objets textes
    texts = []

    # 2. Boucle sur les variables
    for i, feature in enumerate(feature_names):
        x_arrow = loadings[i, pc_x]
        y_arrow = loadings[i, pc_y]
        
        # On ne dessine les flèches que si elles sont assez longues pour être pertinentes
        # (Ça nettoie le centre du cercle)
        if (x_arrow**2 + y_arrow**2)**0.5 > 0.2: 
            
            # Dessiner la flèche
            plt.arrow(0, 0, x_arrow, y_arrow, 
                      color='r', alpha=0.7, head_width=0.02, length_includes_head=True)
            
            # Créer l'objet texte à la pointe (sans le décalage 1.15 cette fois)
            # On stocke cet objet dans la liste 'texts'
            t = ax.text(x_arrow, y_arrow, feature, 
                        color='black', ha='center', va='center', fontsize=11, weight='bold')
            texts.append(t)

    # 3. Limites et titres
    plt.xlim(-1.3, 1.3) # Un peu plus de marge pour les textes
    plt.ylim(-1.3, 1.3)
    var_expl_x = pca.explained_variance_ratio_[pc_x]*100
    var_expl_y = pca.explained_variance_ratio_[pc_y]*100
    plt.xlabel(f'PC{pc_x+1} ({var_expl_x:.1f}%)', fontsize=12)
    plt.ylabel(f'PC{pc_y+1} ({var_expl_y:.1f}%)', fontsize=12)
    plt.title(f'Cercle des Corrélations Lisible (PC{pc_x+1} vs PC{pc_y+1})', 
              fontweight='bold', fontsize=14)
    plt.grid(alpha=0.2)

    # 4. LA MAGIE OPÈRE ICI
    print("Optimisation du placement des labels en cours...")
    adjust_text(texts, 
                # Options pour guider l'algorithme :
                # Relier le texte déplacé à son point d'origine par une ligne grise fine
                arrowprops=dict(arrowstyle='-', color='grey', lw=0.8),
                # Essayer d'éloigner les textes du centre
                expand_points=(1.2, 1.2),
                ax=ax
               )

    plt.show()    


def plot_scree_plot(pca):
    expl_var_ratio = pca.explained_variance_ratio_ * 100
    cum_expl_var_ratio = np.cumsum(expl_var_ratio)
    n_components = len(expl_var_ratio)
    x_axis = np.arange(1, n_components + 1)

    plt.figure(figsize=(10, 6))
    
    # Barres individuelles
    plt.bar(x_axis, expl_var_ratio, alpha=0.6, label='Variance Individuelle', color='royalblue')
    
    # Ligne cumulée
    plt.plot(x_axis, cum_expl_var_ratio, marker='o', linestyle='--', color='orange', linewidth=2, label='Variance Cumulée')
    
    # Seuil indicatif de 80%
    plt.axhline(y=95, color='grey', linestyle=':', label='Seuil 95%')

    plt.xticks(x_axis, [f'PC{i}' for i in x_axis])
    plt.ylabel('Pourcentage de Variance Expliquée (%)')
    plt.xlabel('Composantes Principales')
    plt.title('Scree Plot (Graphique des Éboulis)', fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()


def plot_pca(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test.values, label='Actual (y_test)', alpha=0.7, linewidth=0.8)
    plt.plot(y_test.index, y_pred, label='Predicted (y_pred)', alpha=0.7, linewidth=0.8)
    plt.xlabel('Time')
    plt.ylabel('Load Factor')
    plt.title('Actual vs Predicted Load Factor on Test Set')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_lasso_prediction(y_test, y_pred_l1):
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test.values, label='Réel (2023)', color='black', alpha=0.8, linewidth=2)
    plt.plot(y_test.index, y_pred_l1, label='Lasso (L1)', color='blue', alpha=0.7, linestyle='--')
    plt.title("Réel vs Prédiction Lasso (L1) - Janvier 2023")
    plt.xlabel("Date")
    plt.ylabel("Load Factor")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()    


def plot_xgboost(y_test, y_pred_xgb, model_xgb):
    # Affichage de l'importance des variables
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model_xgb, max_num_features=15, height=0.5)
    plt.title("Importance des Variables (Feature Importance)")
    plt.show()

    # Visualisation Prédiction vs Réalité sur tout le jeu de test
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test.values, label='Réel (2023)', color='black', alpha=0.6)
    plt.plot(y_test.index, y_pred_xgb, label='XGBoost', color='red', alpha=0.6, linestyle='--')
    plt.title("Prédiction Solaire : Réel vs XGBoost (2023 complet)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Visualisation Prédiction vs Réalité
    plt.figure(figsize=(15, 6))
    # On affiche seulement le premier mois de 2023 pour y voir clair
    limit = 24 * 30 
    plt.plot(y_test.index[:limit], y_test.values[:limit], label='Réel (2023)', color='black', alpha=0.7)
    plt.plot(y_test.index[:limit], y_pred_xgb[:limit], label='XGBoost', color='red', alpha=0.7, linestyle='--')
    plt.title("Prédiction Solaire : Réel vs XGBoost (Janvier 2023)")
    plt.legend()
    plt.show()


def plot_rf(y_test,y_pred_rf, model_rf,x_train):
    # Affichage de l'importance des variables
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': x_train.columns,
        'importance': model_rf.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.barh(feature_importance['feature'].head(15), feature_importance['importance'].head(15))
    plt.xlabel('Importance')
    plt.title("Importance des Variables (Feature Importance) - Random Forest")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Visualisation Prédiction vs Réalité sur tout le jeu de test
    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, y_test.values, label='Réel (2023)', color='black', alpha=0.6)
    plt.plot(y_test.index, y_pred_rf, label='Random Forest', color='green', alpha=0.6, linestyle='--')
    plt.title("Prédiction Solaire : Réel vs Random Forest (2023 complet)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Visualisation Prédiction vs Réalité
    plt.figure(figsize=(15, 6))
    # On affiche seulement le premier mois de 2023 pour y voir clair
    limit = 24 * 30 
    plt.plot(y_test.index[:limit], y_test.values[:limit], label='Réel (2023)', color='black', alpha=0.7)
    plt.plot(y_test.index[:limit], y_pred_rf[:limit], label='Random Forest', color='green', alpha=0.7, linestyle='--')
    plt.title("Prédiction Solaire : Réel vs Random Forest (Janvier 2023)")
    plt.legend()
    plt.show()

def compare_models(y_test, y_pred_xgb, y_pred_rf, test_results_df, y_pred, mse_rf, mae_rf):
    # Tableau comparatif des performances

    comparison = pd.DataFrame({
        'Modèle': ['XGBoost', 'Random Forest'],
        'MSE': [mean_squared_error(y_test, y_pred_xgb), mse_rf],
        'MAE': [mean_absolute_error(y_test, y_pred_xgb), mae_rf]
    })


    # Add R² scores to comparison for all models
    comparison['R²'] = [r2_score(y_test, y_pred_xgb), r2_score(y_test, y_pred_rf)]

    # Include regularized models and PCA results

    all_models_comparison = pd.DataFrame({
        'Modèle': ['Ridge (L2)', 'Lasso (L1)', 'ElasticNet', 'PCA', 'XGBoost', 'Random Forest'],
        'MSE': [
            test_results_df.loc[test_results_df['Modèle'] == 'Ridge (L2)', 'MSE'].values[0],
            test_results_df.loc[test_results_df['Modèle'] == 'Lasso (L1)', 'MSE'].values[0],
            test_results_df.loc[test_results_df['Modèle'] == 'ElasticNet', 'MSE'].values[0],
            mean_squared_error(y_test, y_pred),  # PCA predictions from CELL INDEX 0
            mean_squared_error(y_test, y_pred_xgb),
            mse_rf
        ],
        'MAE': [
            test_results_df.loc[test_results_df['Modèle'] == 'Ridge (L2)', 'MAE'].values[0],
            test_results_df.loc[test_results_df['Modèle'] == 'Lasso (L1)', 'MAE'].values[0],
            test_results_df.loc[test_results_df['Modèle'] == 'ElasticNet', 'MAE'].values[0],
            mean_absolute_error(y_test, y_pred),  # PCA predictions from CELL INDEX 0
            mean_absolute_error(y_test, y_pred_xgb),
            mae_rf
        ],
        'R²': [
            test_results_df.loc[test_results_df['Modèle'] == 'Ridge (L2)', 'R²'].values[0],
            test_results_df.loc[test_results_df['Modèle'] == 'Lasso (L1)', 'R²'].values[0],
            test_results_df.loc[test_results_df['Modèle'] == 'ElasticNet', 'R²'].values[0],
            r2_score(y_test, y_pred),  # PCA predictions from CELL INDEX 0
            r2_score(y_test, y_pred_xgb),
            r2_score(y_test, y_pred_rf)
        ]
    })
    print("\n" + "="*70)
    print("COMPARAISON COMPLÈTE DE TOUS LES MODÈLES")
    print("="*70)
    print(all_models_comparison.to_string(index=False))
    print("="*70)

    # Visualisation comparative
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    models_names = all_models_comparison['Modèle']
    colors = ['#FF6B6B', '#FF6B6B', '#FF6B6B', '#4ECDC4', '#E74C3C', '#27AE60']

    axes[0].barh(models_names, all_models_comparison['MSE'], color=colors, alpha=0.8)
    axes[0].set_xlabel('MSE')
    axes[0].set_title('Mean Squared Error')
    axes[0].grid(axis='x', alpha=0.3)

    axes[1].barh(models_names, all_models_comparison['MAE'], color=colors, alpha=0.8)
    axes[1].set_xlabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].grid(axis='x', alpha=0.3)

    axes[2].barh(models_names, all_models_comparison['R²'], color=colors, alpha=0.8)
    axes[2].set_xlabel('R²')
    axes[2].set_title('R² Score (plus élevé = mieux)')
    axes[2].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.show()    