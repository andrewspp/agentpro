import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Configuration du style des graphiques
plt.style.use('whitegrid')
sns.set_palette("viridis") # Utilisation d'une palette de couleurs attrayante


# Dictionnaire des noms de colonnes pour faciliter l'accès
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

##################################################
# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES     #
##################################################

# Chargement des données
file_path = '/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv')
# 🔍 Sélection des colonnes numériques pour éviter les erreurs sur df_numeric.corr()
df_numeric = df.select_dtypes(include='number')


# Affichage des premières lignes pour vérification
print("Aperçu des données initiales:")
print(df.head())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(df.describe())

# Gestion des valeurs manquantes (imputation par la moyenne pour les colonnes numériques)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        df[column].fillna(df[column].mean(), inplace=True)

# Vérification des valeurs manquantes après imputation
print("\nNombre de valeurs manquantes par colonne après imputation:")
print(df.isnull().sum())

# Gestion des outliers (exemple simple: suppression des valeurs hors de 3 écarts-types de la moyenne)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        mean = df[column].mean()
        std = df[column].std()
        df = df[(df[column] >= mean - 3*std) & (df[column] <= mean + 3*std)]

# Affichage des statistiques descriptives après nettoyage
print("\nStatistiques descriptives après nettoyage et imputation:")
print(df.describe())

##################################################
# 2. VISUALISATIONS                               #
##################################################

# 2.1. Matrice de corrélation
corr_matrix = df_numeric.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation des Variables")
plt.savefig("correlation_matrix.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure
correlation_data = corr_matrix # Sauvegarde des données de corrélation

# 2.2. Distribution des scores aux tests
plt.figure(figsize=(8, 6))
sns.histplot(df[col["score_tests"]], kde=True)
plt.title("Distribution des Scores aux Tests")
plt.xlabel("Score aux Tests")
plt.ylabel("Fréquence")
plt.savefig("score_distribution.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure
score_distribution_data = df[col["score_tests"]] # Sauvegarde des données de distribution des scores

# 2.3. Relation entre le budget d'éducation et les scores aux tests
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df[col["log_budget"]], y=df[col["score_tests"]])
plt.title("Relation entre Log du Budget d'Éducation et Scores aux Tests")
plt.xlabel("Log du Budget d'Éducation")
plt.ylabel("Score aux Tests")
plt.savefig("budget_vs_score.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure
budget_score_data = df[[col["log_budget"], col["score_tests"]]] # Sauvegarde des données

# 2.4. Évolution des scores aux tests avant et après la réforme par groupe
# Aggrégation des données
did_data = df.groupby([col["periode"], col["groupe"]])[col["score_tests"]].mean().reset_index()

# Création du graphique
plt.figure(figsize=(10, 6))
sns.lineplot(x=col["periode"], y=col["score_tests"], hue=col["groupe"], data=did_data)
plt.title("Évolution des Scores aux Tests Avant et Après la Réforme")
plt.xlabel("Période")
plt.ylabel("Score aux Tests Moyen")
plt.xticks(did_data[col["periode"]].unique())
plt.legend(title="Groupe")
plt.savefig("did_plot.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure
evolution_data = did_data # Sauvegarde des données d'évolution

# 2.5. Boîte à moustaches du taux d'emploi des jeunes par type d'établissement
plt.figure(figsize=(10, 6))
sns.boxplot(x=col["type_etablissement"], y=col["taux_emploi_jeunes"], data=df)
plt.title("Distribution du Taux d'Emploi des Jeunes par Type d'Établissement")
plt.xlabel("Type d'Établissement")
plt.ylabel("Taux d'Emploi des Jeunes")
plt.xticks(rotation=45)
plt.savefig("emploi_par_etablissement.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure
emploi_etablissement_data = df[[col["type_etablissement"], col["taux_emploi_jeunes"]]] # Sauvegarde des données

##################################################
# 3. MODÉLISATION                               #
##################################################

# 3.1. Préparation des variables
df['reforme_post'] = df[col["reforme"]] * df[col["post"]]  # Variable d'interaction DiD

# 3.2. Modèle DiD de base pour les scores aux tests
formula_scores = f"{col['score_tests']} ~ {col['reforme']} + {col['post']} + reforme_post + {col['log_budget']} + {col['ratio_eleves_enseignant']} + {col['taux_pauvrete']} + {col['niveau_urbanisation']} + C({col['etablissement_id']}) + C({col['periode']})"
model_scores = smf.ols(formula_scores, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[col["etablissement_id"]]})
print("\nRésultats du modèle DiD pour les scores aux tests:")
print(model_scores.summary())
scores_model_results = model_scores.summary() # Sauvegarde des résultats du modèle

# 3.3. Modèle DiD de base pour le taux d'emploi des jeunes
formula_emploi = f"{col['taux_emploi_jeunes']} ~ {col['reforme']} + {col['post']} + reforme_post + {col['log_budget']} + {col['ratio_eleves_enseignant']} + {col['taux_pauvrete']} + {col['niveau_urbanisation']} + C({col['etablissement_id']}) + C({col['periode']})"
model_emploi = smf.ols(formula_emploi, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[col["etablissement_id"]]})
print("\nRésultats du modèle DiD pour le taux d'emploi des jeunes:")
print(model_emploi.summary())
emploi_model_results = model_emploi.summary() # Sauvegarde des résultats du modèle

##################################################
# 4. TESTS DE BASE                              #
##################################################

# 4.1. Analyse des résidus (scores aux tests)
plt.figure(figsize=(8, 6))
sns.residplot(x=model_scores.fittedvalues, y=model_scores.resid, lowess=True)
plt.title("Analyse des Résidus du Modèle (Scores aux Tests)")
plt.xlabel("Valeurs Prédites")
plt.ylabel("Résidus")
plt.savefig("residues_scores.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure
scores_residues_data = pd.DataFrame({'fitted_values': model_scores.fittedvalues, 'residues': model_scores.resid}) # Sauvegarde des données

# 4.2. Analyse des résidus (taux d'emploi des jeunes)
plt.figure(figsize=(8, 6))
sns.residplot(x=model_emploi.fittedvalues, y=model_emploi.resid, lowess=True)
plt.title("Analyse des Résidus du Modèle (Taux d'Emploi des Jeunes)")
plt.xlabel("Valeurs Prédites")
plt.ylabel("Résidus")
plt.savefig("residues_emploi.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure
emploi_residues_data = pd.DataFrame({'fitted_values': model_emploi.fittedvalues, 'residues': model_emploi.resid}) # Sauvegarde des données

# 4.3. Test de multicolinéarité (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Sélection des variables de contrôle pour le test de multicolinéarité
vif_data = df[[col["log_budget"], col["ratio_eleves_enseignant"], col["taux_pauvrete"], col["niveau_urbanisation"]]].dropna()
vif = pd.DataFrame()
vif["Variable"] = vif_data.columns
vif["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

print("\nFacteurs d'Inflation de la Variance (VIF):")
print(vif)
vif_results = vif # Sauvegarde des résultats du VIF