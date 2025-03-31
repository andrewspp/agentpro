import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Configuration du style Seaborn (utiliser 'whitegrid' au lieu de 'whitegrid')
sns.set_style("whitegrid")

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
try:
    # Chargement des données
    df = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv')
    print("Fichier CSV chargé avec succès.")
except FileNotFoundError:
    print("Erreur: Le fichier CSV n'a pas été trouvé. Veuillez vérifier le chemin d'accès.")
    exit()

# Conversion de la colonne 'date' au format datetime
df['date'] = pd.to_datetime(df['date'])

# Gestion des valeurs manquantes (imputation par la médiane pour les variables numériques)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        median_value = df[column].median()
        df[column].fillna(median_value, inplace=True)
print("Valeurs manquantes imputées avec la médiane.")

# Gestion des valeurs aberrantes (méthode IQR pour les variables numériques)
def remove_outliers_iqr(data, column, threshold=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
for column in numeric_columns:
    df = remove_outliers_iqr(df, column)
print("Valeurs aberrantes gérées avec la méthode IQR.")

# Statistiques descriptives
print("\nStatistiques descriptives:\n", df.describe())

# 2. VISUALISATIONS

# A) Matrice de corrélation
correlation_data = df[numeric_columns]
corr_matrix = correlation_data.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation des Variables Numériques")
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
print("Matrice de corrélation créée et sauvegardée.")

# B) Distributions des variables principales
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

sns.histplot(df[col['score_tests']], kde=True, ax=axes[0, 0], color="skyblue")
axes[0, 0].set_title("Distribution des Scores aux Tests")
axes[0, 0].set_xlabel("Score aux Tests")
axes[0, 0].set_ylabel("Fréquence")

sns.histplot(df[col['taux_emploi_jeunes']], kde=True, ax=axes[0, 1], color="lightgreen")
axes[0, 1].set_title("Distribution du Taux d'Emploi des Jeunes")
axes[0, 1].set_xlabel("Taux d'Emploi des Jeunes")
axes[0, 1].set_ylabel("Fréquence")

sns.histplot(df[col['budget_education']], kde=True, ax=axes[1, 0], color="salmon")
axes[1, 0].set_title("Distribution du Budget de l'Éducation")
axes[1, 0].set_xlabel("Budget de l'Éducation")
axes[1, 0].set_ylabel("Fréquence")

sns.histplot(df[col['ratio_eleves_enseignant']], kde=True, ax=axes[1, 1], color="gold")
axes[1, 1].set_title("Distribution du Ratio Élèves/Enseignant")
axes[1, 1].set_xlabel("Ratio Élèves/Enseignant")
axes[1, 1].set_ylabel("Fréquence")

plt.tight_layout()
plt.savefig("variable_distributions.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
print("Distributions des variables principales créées et sauvegardées.")

# C) Relations entre les variables (Scatter plots)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(x=col['budget_education'], y=col['score_tests'], data=df, ax=axes[0], color="coral")
axes[0].set_title("Relation entre Budget de l'Éducation et Scores aux Tests")
axes[0].set_xlabel("Budget de l'Éducation")
axes[0].set_ylabel("Score aux Tests")

sns.scatterplot(x=col['ratio_eleves_enseignant'], y=col['score_tests'], data=df, ax=axes[1], color="mediumpurple")
axes[1].set_title("Relation entre Ratio Élèves/Enseignant et Scores aux Tests")
axes[1].set_xlabel("Ratio Élèves/Enseignant")
axes[1].set_ylabel("Score aux Tests")

plt.tight_layout()
plt.savefig("variable_relationships.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
print("Relations entre les variables créées et sauvegardées.")

# D) DiD - Graphique de tendance temporelle (score_tests)
# Créer un DataFrame pour le graphique de tendance temporelle
trend_data = df.groupby(['annee', 'groupe'])[col['score_tests']].mean().reset_index()
trend_data['annee'] = pd.to_datetime(trend_data['annee'], format='%Y')

# Créer le graphique
plt.figure(figsize=(12, 6))
sns.lineplot(x='annee', y=col['score_tests'], hue='groupe', data=trend_data, marker='o')
plt.title('Évolution des Scores aux Tests par Groupe (Réformé vs Non Réformé)')
plt.xlabel('Année')
plt.ylabel('Score Moyen aux Tests')
plt.legend(title='Groupe')
plt.grid(True)

# Ajouter une ligne verticale pour indiquer le moment de la réforme
plt.axvline(pd.to_datetime('2016', format='%Y'), color='red', linestyle='--', label='Début de la Réforme')
plt.legend()
plt.tight_layout()
plt.savefig("did_trend_score_tests.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

print("Graphique de tendance temporelle DiD (Score Tests) créé et sauvegardé.")

# 3. MODÉLISATION

# A) Préparation des données pour la régression
# Convertir les variables catégorielles en variables muettes
df = pd.get_dummies(df, columns=[col['type_etablissement'], col['approche_pedagogique'], col['groupe']])

# B) Modèle DiD pour Score_Tests
formula_score_tests = (
    f"{col['score_tests']} ~ {col['reforme']} + {col['post']} + {col['interaction_did']} + "
    f"{col['budget_education']} + {col['ratio_eleves_enseignant']} + {col['taux_pauvrete']} + "
    f"{col['niveau_urbanisation']} + C({col['etablissement_id']}) + C({col['periode']})"  # Effets fixes
)

model_score_tests = smf.ols(formula_score_tests, data=df).fit()
print("\nRésultats du modèle DiD (Score Tests):\n")
print(model_score_tests.summary())

# C) Modèle DiD pour Taux_Emploi_Jeunes
formula_taux_emploi = (
    f"{col['taux_emploi_jeunes']} ~ {col['reforme']} + {col['post']} + {col['interaction_did']} + "
    f"{col['budget_education']} + {col['ratio_eleves_enseignant']} + {col['taux_pauvrete']} + "
    f"{col['niveau_urbanisation']} + C({col['etablissement_id']}) + C({col['periode']})"  # Effets fixes
)

model_taux_emploi = smf.ols(formula_taux_emploi, data=df).fit()
print("\nRésultats du modèle DiD (Taux Emploi Jeunes):\n")
print(model_taux_emploi.summary())

# 4. TESTS DE BASE

# A) Analyse des résidus (Score Tests)
plt.figure(figsize=(10, 6))
sns.residplot(x=model_score_tests.fittedvalues, y=model_score_tests.resid, lowess=True, color="darkred")
plt.title("Analyse des Résidus (Score Tests)")
plt.xlabel("Valeurs Prédites")
plt.ylabel("Résidus")
plt.savefig("resid_analysis_score_tests.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

print("Analyse des résidus (Score Tests) effectuée et sauvegardée.")

# B) Test de multicolinéarité (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Préparation des variables indépendantes pour le VIF
independent_vars = sm.add_constant(df[[col['budget_education'], col['ratio_eleves_enseignant'], col['taux_pauvrete'], col['niveau_urbanisation']]])

# Calcul des VIFs
vif_data = pd.DataFrame()
vif_data["Variable"] = independent_vars.columns
vif_data["VIF"] = [variance_inflation_factor(independent_vars.values, i) for i in range(independent_vars.shape[1])]

print("\nFacteurs d'Inflation de la Variance (VIF):\n", vif_data)