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
├── requirements.txt       # Dépendances
└── README.md              # Ce fichier
