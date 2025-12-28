# Prévision de Production Photovoltaïque : Approche Stochastique
## Cas d'Étude : Région Parisienne (Île-de-France)

Ce projet propose une chaîne de traitement complète (pipeline) pour la modélisation et la prévision à court terme de la production d'énergie solaire. Il est conçu pour être reproductible et simple d'exécution.

### 1. Problématique Scientifique

L'intégration des énergies renouvelables (EnR) dans le réseau électrique (Grid) pose un problème d'optimisation sous incertitude. La production photovoltaïque $Y_t$ est un processus stochastique non-stationnaire dépendant de variables exogènes météorologiques $X_t$.

L'objectif est d'estimer la fonction de transfert $f$ telle que :  
```math
\hat{Y}_{t+h} = f(Y_{t}, Y_{t-1}, \dots, X_{t+h}) + \epsilon_t
```
Où :
* $Y_t$ est la production réelle en MW (Source : RTE, périmètre Île-de-France).
* $X_t$ est le vecteur d'état météorologique à Paris (GHI, Température, Nébulosité).
* $\epsilon_t$ est le terme d'erreur que nous cherchons à minimiser (RMSE).

### 2. Architecture du Projet

Le projet est structuré pour une exécution linéaire via un unique point d'entrée.

```text
solar-forecasting-paris/
├── data/                  # Dossier de stockage (géré automatiquement)
├── src/                   # Modules Python (Backend scientifique)
│   ├── data_loader.py     # Clients API (Open-Meteo & ODRÉ)
│   ├── processing.py      # Nettoyage et Feature Engineering
│   └── modeling.py        # Définition des modèles (XGBoost/LSTM)
├── main.ipynb             # LE notebook d'exécution unique
├── utils.py               # Les outils que nous allons appeler dans le main
└── README.md              # Ce fichier


## 3. Pipeline de Traitement

### 3.1. Acquisition des Données
* **Production** : RTE via ODRÉ (Île-de-France, 30min, 2018-2023)
* **Météo** : Open-Meteo API (GHI, DNI, DHI, T, Vent, Cloud) - Paris, Marseille, Bordeaux
* **Fusion** : Jointure temporelle

### 3.2. Analyse Exploratoire (EDA)
* Visualisations temporelles (cycles diurnes/saisonniers)
* Statistiques descriptives et distributions
* Détection de la non-stationnarité

### 3.3. Prétraitement
* **Capacité installée** : Récupération via API ODRÉ
* **Load Factor** : `Production / Capacité` (normalisation 0-1)
* **Nettoyage** : Correction des valeurs aberrantes

### 3.4. Feature Engineering
* **Target** : Production à t+1h
* **Lags** : [30min, 1h, 3h, 6h, 12h, 24h] sur production et météo
* **Rolling stats** : mean, std, max, min sur [1h, 3h, 24h]
* **Calendaires** : hour, day_of_week, month, day_of_year

### 3.5. Modélisation
* **Split temporel** : 80% train / 20% test (pas de shuffle)
* **Baseline** : Modèle de persistance
* **Modèles testés** :
  - Régression Linéaire
  - PCA + Régression
  - **XGBoost** (n=1000, lr=0.05, depth=6)
  - **Random Forest** (n=1000, depth=15)

### 3.6. Évaluation
* **Métriques** : RMSE, MAE
* **Feature importance** : Top 15 variables
* **Visualisations** : Réel vs Prédit (année complète + zoom)
* **Comparaison** : Performance XGBoost vs Random Forest