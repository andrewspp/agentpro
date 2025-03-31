import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration du style seaborn (IMPORTANT : utilisation de 'whitegrid')
sns.set_theme(style='whitegrid')

# Dictionnaire des noms de colonnes
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
file_path = '/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv')
# 🔍 Sélection des colonnes numériques pour éviter les erreurs sur df_numeric.corr()
df_numeric = df.select_dtypes(include='number')


# Afficher les premières lignes du DataFrame
print("Aperçu du DataFrame initial :")
print(df.head())

# Statistiques descriptives
print("\nStatistiques descriptives initiales :")
print(df.describe())

# Gestion des valeurs manquantes
print("\nNombre de valeurs manquantes par colonne :")
print(df.isnull().sum())

# Imputation des valeurs manquantes (exemple avec la moyenne)
for column in df.columns:
    if df[column].isnull().any():
        if df[column].dtype in ['int64', 'float64']:
            df[column].fillna(df[column].mean(), inplace=True) #Imputation par la moyenne pour les numériques
        else:
            df[column].fillna(df[column].mode()[0], inplace=True) #Imputation par le mode pour les objets (chaînes de caractères)

print("\nNombre de valeurs manquantes après imputation :")
print(df.isnull().sum())

# Gestion des outliers (exemple simple: suppression des outliers sur 'nb_eleves')
Q1 = df[col["nb_eleves"]].quantile(0.25)
Q3 = df[col["nb_eleves"]].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df[(df[col["nb_eleves"]] >= lower_bound) & (df[col["nb_eleves"]] <= upper_bound)]

print("\nTaille du DataFrame avant suppression des outliers:", len(df))
print("Taille du DataFrame après suppression des outliers:", len(df_no_outliers))

#On travaille désormais avec le dataframe sans outliers
df = df_no_outliers.copy()

# 2. VISUALISATIONS
# Matrice de corrélation
correlation_matrix = df_numeric.corr(numeric_only=True)
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation")
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

correlation_data = correlation_matrix  # Stockage des données de la matrice

# Distributions des variables principales
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.histplot(df[col["score_tests"]], kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title("Distribution des Scores aux Tests")
axes[0, 0].set_xlabel("Score aux Tests")
axes[0, 0].set_ylabel("Fréquence")

sns.histplot(df[col["taux_emploi_jeunes"]], kde=True, ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title("Distribution du Taux d'Emploi des Jeunes")
axes[0, 1].set_xlabel("Taux d'Emploi des Jeunes")
axes[0, 1].set_ylabel("Fréquence")

sns.histplot(df[col["budget_education"]], kde=True, ax=axes[1, 0], color='lightcoral')
axes[1, 0].set_title("Distribution du Budget de l'Education")
axes[1, 0].set_xlabel("Budget de l'Education")
axes[1, 0].set_ylabel("Fréquence")

sns.histplot(df[col["ratio_eleves_enseignant"]], kde=True, ax=axes[1, 1], color='gold')
axes[1, 1].set_title("Distribution du Ratio Eleves/Enseignant")
axes[1, 1].set_xlabel("Ratio Eleves/Enseignant")
axes[1, 1].set_ylabel("Fréquence")

plt.tight_layout()
plt.savefig("distributions_principales.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

distributions_data = df[[col["score_tests"], col["taux_emploi_jeunes"], col["budget_education"], col["ratio_eleves_enseignant"]]].copy()  # Stockage des données

# Relation entre score aux tests et taux d'emploi des jeunes
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[col["score_tests"]], y=df[col["taux_emploi_jeunes"]], alpha=0.7, color='purple')
plt.title("Relation entre Score aux Tests et Taux d'Emploi des Jeunes")
plt.xlabel("Score aux Tests")
plt.ylabel("Taux d'Emploi des Jeunes")
plt.savefig("relation_score_emploi.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

relation_data = df[[col["score_tests"], col["taux_emploi_jeunes"]]].copy()  # Stockage des données

# Boxplots par type d'établissement
plt.figure(figsize=(12, 6))
sns.boxplot(x=df[col["type_etablissement"]], y=df[col["score_tests"]], color='orange')
plt.title("Boxplots des Scores aux Tests par Type d'Etablissement")
plt.xlabel("Type d'Etablissement")
plt.ylabel("Score aux Tests")
plt.savefig("boxplots_type_etablissement.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

boxplots_data = df[[col["type_etablissement"], col["score_tests"]]].copy()  # Stockage des données

# Parallel trends assumption
plt.figure(figsize=(12, 6))
sns.lineplot(x=df[col["periode"]], y=df[col["score_tests"]], hue=df[col["reforme"]].astype(str), marker='o')
plt.axvline(x=2, color='red', linestyle='--', label='Réforme implémentée')
plt.title("Vérification de l'hypothèse de tendances parallèles")
plt.xlabel("Période")
plt.ylabel("Score aux Tests")
plt.legend(title="Réforme")
plt.savefig("parallel_trends.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

parallel_trends_data = df[[col["periode"], col["score_tests"], col["reforme"]]].copy() # Stockage des données

# 3. MODÉLISATION (Difference-in-Differences)
# Regression DiD pour l'impact sur les scores aux tests
formula_score = f"{col['score_tests']} ~ {col['reforme']} + {col['post']} + {col['interaction_did']} + {col['log_budget']} + {col['ratio_eleves_enseignant']} + {col['taux_pauvrete']} + {col['niveau_urbanisation']}"
model_score = smf.ols(formula_score, data=df)
results_score = model_score.fit()
print("\nRésultats de la régression DiD (Score aux Tests):\n", results_score.summary())

# Regression DiD pour l'impact sur le taux d'emploi des jeunes
formula_emploi = f"{col['taux_emploi_jeunes']} ~ {col['reforme']} + {col['post']} + {col['interaction_did']} + {col['log_budget']} + {col['ratio_eleves_enseignant']} + {col['taux_pauvrete']} + {col['niveau_urbanisation']}"
model_emploi = smf.ols(formula_emploi, data=df)
results_emploi = model_emploi.fit()
print("\nRésultats de la régression DiD (Taux d'Emploi des Jeunes):\n", results_emploi.summary())

# 4. TESTS DE BASE
# Analyse des résidus (Score aux Tests)
plt.figure(figsize=(10, 6))
sns.residplot(x=results_score.fittedvalues, y=results_score.resid, lowess=True, color="darkblue")
plt.title("Analyse des Résidus (Score aux Tests)")
plt.xlabel("Valeurs Prédites")
plt.ylabel("Résidus")
plt.savefig("residues_score_tests.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

residues_score_data = pd.DataFrame({'fitted_values': results_score.fittedvalues, 'resid': results_score.resid}) # Stockage des données

# Analyse des résidus (Taux d'Emploi des Jeunes)
plt.figure(figsize=(10, 6))
sns.residplot(x=results_emploi.fittedvalues, y=results_emploi.resid, lowess=True, color="darkgreen")
plt.title("Analyse des Résidus (Taux d'Emploi des Jeunes)")
plt.xlabel("Valeurs Prédites")
plt.ylabel("Résidus")
plt.savefig("residues_taux_emploi.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

residues_emploi_data = pd.DataFrame({'fitted_values': results_emploi.fittedvalues, 'resid': results_emploi.resid}) # Stockage des données

# Test de multicolinéarité (VIF)
def calculate_vif(data):
    vif = pd.DataFrame()
    vif["variables"] = data.columns
    vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif

# Sélection des variables pour le test de multicolinéarité
variables_for_vif = df[[col["reforme"], col["post"], col["interaction_did"], col["log_budget"], col["ratio_eleves_enseignant"], col["taux_pauvrete"], col["niveau_urbanisation"]]]

vif_results = calculate_vif(variables_for_vif)
print("\nRésultats du test de multicolinéarité (VIF):\n", vif_results)

