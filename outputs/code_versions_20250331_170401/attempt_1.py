import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# Configuration du style Seaborn (important pour la compatibilité)
sns.set_theme(style="whitegrid")

# Définition du dictionnaire des noms de colonnes
col = {
    "etablissement_id": "etablissement_id",
    "type_etablissement": "type_etablissement",
    "periode": "periode",
    "date": "date",
    "annee": "annee",
    "semestre": "semestre",
    "reforme": "reforme",
    "post": "post",
    "interaction_did": "interaction_did",
    "budget_education": "budget_education",
    "nb_eleves": "nb_eleves",
    "ratio_eleves_enseignant": "ratio_eleves_enseignant",
    "taux_pauvrete": "taux_pauvrete",
    "niveau_urbanisation": "niveau_urbanisation",
    "approche_pedagogique": "approche_pedagogique",
    "score_tests": "score_tests",
    "taux_emploi_jeunes": "taux_emploi_jeunes",
    "log_budget": "log_budget",
    "log_nb_eleves": "log_nb_eleves",
    "groupe": "groupe",
    "periode_relative": "periode_relative",
    "phase": "phase"
}

# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# Chargement des données à partir du fichier CSV
file_path = '/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv')

# Affichage des premières lignes du DataFrame pour vérification
print("Aperçu du DataFrame:")
print(df.head())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(df.describe())

# Gestion des valeurs manquantes
# Identification des colonnes avec des valeurs manquantes
missing_values = df.isnull().sum()
print("\nNombre de valeurs manquantes par colonne:")
print(missing_values[missing_values > 0])

# Imputation des valeurs manquantes (exemple : remplacement par la moyenne)
for column in df.columns:
    if df[column].isnull().any():
        if df[column].dtype == 'float64' or df[column].dtype == 'int64':
            df[column].fillna(df[column].mean(), inplace=True)
        else:
            df[column].fillna(df[column].mode()[0], inplace=True)

# Vérification de l'absence de valeurs manquantes après imputation
print("\nNombre de valeurs manquantes après imputation:")
print(df.isnull().sum().sum())

# Gestion des outliers (exemple : suppression des valeurs hors de 3 écarts-types de la moyenne)
for column in [col["score_tests"], col["taux_emploi_jeunes"], col["budget_education"], col["nb_eleves"], col["ratio_eleves_enseignant"], col["taux_pauvrete"], col["niveau_urbanisation"]]:
    if df[column].dtype == 'float64' or df[column].dtype == 'int64':
        mean = df[column].mean()
        std = df[column].std()
        df = df[(df[column] >= mean - 3*std) & (df[column] <= mean + 3*std)]

# Statistiques descriptives après le nettoyage
print("\nStatistiques descriptives après le nettoyage:")
print(df.describe())

# 2. VISUALISATIONS

# 2.1 Matrice de corrélation
correlation_matrix_data = df[[col["score_tests"], col["taux_emploi_jeunes"], col["budget_education"], col["nb_eleves"], col["ratio_eleves_enseignant"], col["taux_pauvrete"], col["niveau_urbanisation"]]].copy()
corr = correlation_matrix_data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation des Variables Numériques")
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.2 Distributions des variables principales
plt.figure(figsize=(12, 6))
sns.histplot(df[col["score_tests"]], kde=True)
plt.title("Distribution des Scores aux Tests")
plt.xlabel("Score aux Tests")
plt.ylabel("Fréquence")
plt.savefig("score_distribution.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
score_distribution_data = df[col["score_tests"]].copy()  # Stockage des données


plt.figure(figsize=(12, 6))
sns.histplot(df[col["taux_emploi_jeunes"]], kde=True)
plt.title("Distribution du Taux d'Emploi des Jeunes")
plt.xlabel("Taux d'Emploi des Jeunes")
plt.ylabel("Fréquence")
plt.savefig("emploi_distribution.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
emploi_distribution_data = df[col["taux_emploi_jeunes"]].copy() # Stockage des données


# 2.3 Relation entre le score aux tests et le taux d'emploi des jeunes
plt.figure(figsize=(12, 6))
sns.scatterplot(x=df[col["score_tests"]], y=df[col["taux_emploi_jeunes"]])
plt.title("Relation entre le Score aux Tests et le Taux d'Emploi des Jeunes")
plt.xlabel("Score aux Tests")
plt.ylabel("Taux d'Emploi des Jeunes")
plt.savefig("score_emploi_relation.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
score_emploi_relation_data = df[[col["score_tests"], col["taux_emploi_jeunes"]]].copy() # Stockage des données


# 2.4 Graphique de tendance temporelle avec deux groupes avant et après traitement
# Création d'une colonne 'time' pour représenter le temps relatif à la réforme
df['time'] = df[col["annee"]] + (df[col["semestre"]] - 1) * 0.5
time_trend_data = df.groupby(['time', col["reforme"]])[col["score_tests"]].mean().reset_index()
plt.figure(figsize=(14, 7))
sns.lineplot(x='time', y=col["score_tests"], hue=col["reforme"], data=time_trend_data)
plt.axvline(x=2016, color='red', linestyle='--', label='Réforme (2016)')
plt.title('Tendances des scores aux tests avant et après la réforme')
plt.xlabel('Temps')
plt.ylabel('Moyenne des scores aux tests')
plt.legend(title='Réforme')
plt.savefig("time_trend.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 3. MODÉLISATION DiD
# Assurez-vous que les variables catégorielles sont correctement encodées (si nécessaire)
df[col["type_etablissement"]] = df[col["type_etablissement"]].astype('category')
df[col["approche_pedagogique"]] = df[col["approche_pedagogique"]].astype('category')
df[col["groupe"]] = df[col["groupe"]].astype('category')
df[col["phase"]] = df[col["phase"]].astype('category')

# Modèle DiD pour le score aux tests
formula_score = f'{col["score_tests"]} ~ {col["reforme"]} * {col["post"]} + {col["budget_education"]} + {col["ratio_eleves_enseignant"]} + {col["taux_pauvrete"]} + {col["niveau_urbanisation"]} + C({col["type_etablissement"]}) + C({col["approche_pedagogique"]}) + C({col["etablissement_id"]}) + C({col["annee"]})'
model_score = smf.ols(formula_score, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[col["etablissement_id"]]})
print("\nRésultats du modèle DiD pour le score aux tests:")
print(model_score.summary())

# Modèle DiD pour le taux d'emploi des jeunes
formula_emploi = f'{col["taux_emploi_jeunes"]} ~ {col["reforme"]} * {col["post"]} + {col["budget_education"]} + {col["ratio_eleves_enseignant"]} + {col["taux_pauvrete"]} + {col["niveau_urbanisation"]} + C({col["type_etablissement"]}) + C({col["approche_pedagogique"]}) + C({col["etablissement_id"]}) + C({col["annee"]})'
model_emploi = smf.ols(formula_emploi, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[col["etablissement_id"]]})
print("\nRésultats du modèle DiD pour le taux d'emploi des jeunes:")
print(model_emploi.summary())

# 4. TESTS DE BASE
# Analyse des résidus (exemple pour le modèle du score aux tests)
plt.figure(figsize=(12, 6))
sns.histplot(model_score.resid, kde=True)
plt.title("Distribution des Résidus (Score aux Tests)")
plt.xlabel("Résidus")
plt.ylabel("Fréquence")
plt.savefig("residues_score.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 5. ANALYSE DE L'HYPOTHÈSE DE TENDANCES PARALLÈLES

# Création de variables d'interaction pour chaque période avant la réforme
for year in df[col["annee"]].unique():
    if year < df[col["annee"]].min() + 2:  # Considérer seulement les années avant la réforme
        df[f'Reforme_x_Annee_{year}'] = df[col["reforme"]] * (df[col["annee"]] == year)

# Ajout des variables d'interaction au modèle
parallel_trend_formula_score = f'{col["score_tests"]} ~ {col["reforme"]} * {col["post"]} + {col["budget_education"]} + {col["ratio_eleves_enseignant"]} + {col["taux_pauvrete"]} + {col["niveau_urbanisation"]} + C({col["type_etablissement"]}) + C({col["approche_pedagogique"]}) + C({col["etablissement_id"]}) + C({col["annee"]})'
for year in df[col["annee"]].unique():
    if year < df[col["annee"]].min() + 2:
        parallel_trend_formula_score += f' + Reforme_x_Annee_{year}'

parallel_trend_model_score = smf.ols(parallel_trend_formula_score, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[col["etablissement_id"]]})
print("\nRésultats du modèle avec tests de tendances parallèles pour le score aux tests:")
print(parallel_trend_model_score.summary())

print("Analyse terminée.")