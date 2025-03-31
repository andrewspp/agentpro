import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Configuration du style Seaborn (en utilisant la version la plus récente disponible)
sns.set_theme(style="whitegrid")

# Dictionnaire de colonnes
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
chemin_fichier = '/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv')

# Gestion des valeurs manquantes (imputation par la moyenne pour les numériques)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        df[column].fillna(df[column].mean(), inplace=True)

# Suppression des outliers (méthode simple: on fixe des bornes arbitraires)
# Note: Une analyse plus poussée des outliers serait préférable dans un contexte réel.
for column in [col["budget_education"], col["nb_eleves"], col["ratio_eleves_enseignant"], col["taux_pauvrete"], col["niveau_urbanisation"], col["score_tests"], col["taux_emploi_jeunes"]]:
    lower_bound = df[column].quantile(0.01)
    upper_bound = df[column].quantile(0.99)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Création de statistiques descriptives
description = df.describe(include='all')
print(description)


# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# 2.1 Matrice de corrélation
corr_matrix = df[[col["budget_education"], col["nb_eleves"], col["ratio_eleves_enseignant"], col["taux_pauvrete"], col["niveau_urbanisation"], col["score_tests"], col["taux_emploi_jeunes"]]].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation des Variables Numériques")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.2 Distributions des variables principales
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(df[col["score_tests"]], kde=True, ax=axes[0, 0], color="skyblue")
axes[0, 0].set_title("Distribution des Scores aux Tests")
axes[0, 0].set_xlabel("Score")
axes[0, 0].set_ylabel("Fréquence")

sns.histplot(df[col["taux_emploi_jeunes"]], kde=True, ax=axes[0, 1], color="lightgreen")
axes[0, 1].set_title("Distribution du Taux d'Emploi des Jeunes")
axes[0, 1].set_xlabel("Taux d'Emploi")
axes[0, 1].set_ylabel("Fréquence")

sns.histplot(df[col["budget_education"]], kde=True, ax=axes[1, 0], color="lightcoral")
axes[1, 0].set_title("Distribution du Budget de l'Éducation")
axes[1, 0].set_xlabel("Budget")
axes[1, 0].set_ylabel("Fréquence")

sns.histplot(df[col["ratio_eleves_enseignant"]], kde=True, ax=axes[1, 1], color="gold")
axes[1, 1].set_title("Distribution du Ratio Élèves/Enseignant")
axes[1, 1].set_xlabel("Ratio")
axes[1, 1].set_ylabel("Fréquence")


plt.tight_layout()
plt.savefig("variable_distributions.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.3 Relation entre le budget et les scores aux tests
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[col["budget_education"]], y=df[col["score_tests"]], alpha=0.6, color="mediumpurple")
plt.title("Relation entre le Budget de l'Éducation et les Scores aux Tests")
plt.xlabel("Budget de l'Éducation")
plt.ylabel("Score aux Tests")
plt.tight_layout()
plt.savefig("budget_vs_score.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.4 Boîtes à moustaches par type d'établissement
plt.figure(figsize=(12, 7))
sns.boxplot(x=df[col["type_etablissement"]], y=df[col["score_tests"]], palette="viridis")
plt.title("Distribution des Scores aux Tests par Type d'Établissement")
plt.xlabel("Type d'Établissement")
plt.ylabel("Score aux Tests")
plt.tight_layout()
plt.savefig("boxplot_type_etablissement.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.5 Parallel trends assumption check
# This is a crucial step for Diff-in-Diff analysis
# Create a line plot to check parallel trends before the intervention

# Create a DataFrame to store the data for the parallel trends plot
parallel_trends_data = df.groupby([col["periode"], col["reforme"]])[col["score_tests"]].mean().unstack()

# Plotting the parallel trends
plt.figure(figsize=(12, 6))
plt.plot(parallel_trends_data[0], marker='o', label='Non Réformé')
plt.plot(parallel_trends_data[1], marker='o', label='Réformé')

# Adding a vertical line at the intervention point (periode 2 here)
plt.axvline(x=2, color='red', linestyle='--', label='Réforme (Periode 2)')

plt.title('Vérification des Tendances Parallèles avant la Réforme')
plt.xlabel('Période')
plt.ylabel('Moyenne des Scores aux Tests')
plt.legend()
plt.tight_layout()
plt.savefig("parallel_trends.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()


# 3. MODÉLISATION SIMPLE ET CLAIRE

# Convertir les variables catégorielles en variables dummies
df = pd.get_dummies(df, columns=[col["type_etablissement"], col["approche_pedagogique"]])

# 3.1 Modèle DiD pour l'impact sur les scores aux tests
# 'score_tests ~ interaction_did + reforme + post + budget_education + ratio_eleves_enseignant + taux_pauvrete + niveau_urbanisation'
formula = f'{col["score_tests"]} ~ {col["interaction_did"]} + {col["reforme"]} + {col["post"]} + {col["budget_education"]} + {col["ratio_eleves_enseignant"]} + {col["taux_pauvrete"]} + {col["niveau_urbanisation"]}'
model = smf.ols(formula, data=df).fit()
print("Résultats du modèle DiD (Scores aux Tests):")
print(model.summary())

# 3.2 Modèle DiD pour l'impact sur le taux d'emploi des jeunes
formula_emploi = f'{col["taux_emploi_jeunes"]} ~ {col["interaction_did"]} + {col["reforme"]} + {col["post"]} + {col["budget_education"]} + {col["ratio_eleves_enseignant"]} + {col["taux_pauvrete"]} + {col["niveau_urbanisation"]}'
model_emploi = smf.ols(formula_emploi, data=df).fit()
print("\nRésultats du modèle DiD (Taux d'Emploi des Jeunes):")
print(model_emploi.summary())

# 4. TESTS DE BASE

# 4.1 Analyse des résidus (scores aux tests)
plt.figure(figsize=(10, 6))
sns.residplot(x=df[col["interaction_did"]], y=model.resid, lowess=True, color="darkred")
plt.title("Analyse des Résidus (Scores aux Tests)")
plt.xlabel("Variable d'Interaction (reforme * post)")
plt.ylabel("Résidus")
plt.tight_layout()
plt.savefig("resid_analysis_scores.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 4.2 Analyse des résidus (taux d'emploi des jeunes)
plt.figure(figsize=(10, 6))
sns.residplot(x=df[col["interaction_did"]], y=model_emploi.resid, lowess=True, color="darkblue")
plt.title("Analyse des Résidus (Taux d'Emploi des Jeunes)")
plt.xlabel("Variable d'Interaction (reforme * post)")
plt.ylabel("Résidus")
plt.tight_layout()
plt.savefig("resid_analysis_emploi.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 4.3 Test de multicolinéarité (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Sélectionner les variables indépendantes numériques
X = df[[col["budget_education"], col["ratio_eleves_enseignant"], col["taux_pauvrete"], col["niveau_urbanisation"]]]

# Calculer le VIF pour chaque variable
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nFacteur d'Inflation de Variance (VIF):")
print(vif_data)

# 5. CAPTURE ET STOCKAGE DES DONNÉES POUR INTERPRÉTATION
# Les DataFrames utilisés pour chaque visualisation ont été stockés directement lors de leur création.
# Par exemple:
# - corr_matrix est le DataFrame pour la matrice de corrélation
# - parallel_trends_data est le DataFrame pour la vérification des tendances parallèles

print("Analyse terminée. Les visualisations et les résultats ont été sauvegardés.")