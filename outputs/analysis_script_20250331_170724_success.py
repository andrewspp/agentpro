import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Configuration du style des graphiques
sns.set_theme(style="whitegrid")

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
file_path = '/Users/pierreandrews/Desktop/agentpro/donnees3.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/donnees3.csv')

# Affichage des premières lignes pour vérification
print("Aperçu des premières lignes du DataFrame:")
print(df.head())

# Statistiques descriptives
print("\nStatistiques descriptives du DataFrame:")
print(df.describe())

# Gestion des valeurs manquantes (imputation par la moyenne pour simplifier, mais l'imputation multiple est préférable)
for column in df.columns:
    if df[column].isnull().any():
        df[column].fillna(df[column].mean(), inplace=True)

print("\nVérification des valeurs manquantes après imputation:")
print(df.isnull().sum())

# Gestion des outliers (méthode simple, mais des méthodes plus robustes existent)
for column in ['budget_education', 'nb_eleves', 'ratio_eleves_enseignant', 'taux_pauvrete', 'niveau_urbanisation', 'score_tests', 'taux_emploi_jeunes']:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
    df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])

print("\nPrétraitement des données terminé.")

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES
# Matrice de corrélation
corr_matrix = df[['budget_education', 'nb_eleves', 'ratio_eleves_enseignant', 'taux_pauvrete', 'niveau_urbanisation', 'score_tests', 'taux_emploi_jeunes']].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation des Variables Principales")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

correlation_data = corr_matrix.copy()

# Distributions des variables principales
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
sns.histplot(df[col["score_tests"]], kde=True, ax=axes[0, 0], color="skyblue")
axes[0, 0].set_title("Distribution des Scores aux Tests")
axes[0, 0].set_xlabel("Scores aux Tests")
axes[0, 0].set_ylabel("Fréquence")

sns.histplot(df[col["taux_emploi_jeunes"]], kde=True, ax=axes[0, 1], color="lightgreen")
axes[0, 1].set_title("Distribution du Taux d'Emploi des Jeunes")
axes[0, 1].set_xlabel("Taux d'Emploi des Jeunes")
axes[0, 1].set_ylabel("Fréquence")

sns.histplot(df[col["budget_education"]], kde=True, ax=axes[0, 2], color="lightcoral")
axes[0, 2].set_title("Distribution du Budget d'Éducation")
axes[0, 2].set_xlabel("Budget d'Éducation")
axes[0, 2].set_ylabel("Fréquence")

sns.histplot(df[col["taux_pauvrete"]], kde=True, ax=axes[1, 0], color="lightsalmon")
axes[1, 0].set_title("Distribution du Taux de Pauvreté")
axes[1, 0].set_xlabel("Taux de Pauvreté")
axes[1, 0].set_ylabel("Fréquence")

sns.histplot(df[col["niveau_urbanisation"]], kde=True, ax=axes[1, 1], color="lavender")
axes[1, 1].set_title("Distribution du Niveau d'Urbanisation")
axes[1, 1].set_xlabel("Niveau d'Urbanisation")
axes[1, 1].set_ylabel("Fréquence")

sns.histplot(df[col["ratio_eleves_enseignant"]], kde=True, ax=axes[1, 2], color="gold")
axes[1, 2].set_title("Distribution du Ratio Élèves/Enseignant")
axes[1, 2].set_xlabel("Ratio Élèves/Enseignant")
axes[1, 2].set_ylabel("Fréquence")

plt.tight_layout()
plt.savefig("distributions.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

distributions_data = df[[col["score_tests"], col["taux_emploi_jeunes"], col["budget_education"], col["taux_pauvrete"], col["niveau_urbanisation"], col["ratio_eleves_enseignant"]]].copy()

# Relations entre variables importantes (nuages de points)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.scatterplot(x=col["budget_education"], y=col["score_tests"], data=df, ax=axes[0], color="mediumseagreen")
axes[0].set_title("Relation entre Budget d'Éducation et Scores aux Tests")
axes[0].set_xlabel("Budget d'Éducation")
axes[0].set_ylabel("Scores aux Tests")

sns.scatterplot(x=col["taux_pauvrete"], y=col["taux_emploi_jeunes"], data=df, ax=axes[1], color="coral")
axes[1].set_title("Relation entre Taux de Pauvreté et Taux d'Emploi des Jeunes")
axes[1].set_xlabel("Taux de Pauvreté")
axes[1].set_ylabel("Taux d'Emploi des Jeunes")

plt.tight_layout()
plt.savefig("relationships.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

relationships_data = df[[col["budget_education"], col["score_tests"], col["taux_pauvrete"], col["taux_emploi_jeunes"]]].copy()

# Analyse DiD: Graphique de tendance temporelle
# Création de la variable d'interaction
df['interaction_did'] = df['reforme'] * df['post']

# Moyennes par période et groupe
did_data = df.groupby(['periode', 'reforme'])[['score_tests', 'taux_emploi_jeunes']].mean().reset_index()

# Graphique
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Scores aux tests
sns.lineplot(x='periode', y='score_tests', hue='reforme', data=did_data, marker='o', ax=axes[0])
axes[0].set_title('Évolution des Scores aux Tests (DiD)')
axes[0].set_xlabel('Période')
axes[0].set_ylabel('Score moyen aux tests')
axes[0].legend(title='Réforme', labels=['Groupe Contrôle', 'Groupe Traité'])

# Taux d'emploi des jeunes
sns.lineplot(x='periode', y='taux_emploi_jeunes', hue='reforme', data=did_data, marker='o', ax=axes[1])
axes[1].set_title('Évolution du Taux d\'Emploi des Jeunes (DiD)')
axes[1].set_xlabel('Période')
axes[1].set_ylabel('Taux d\'emploi moyen des jeunes')
axes[1].legend(title='Réforme', labels=['Groupe Contrôle', 'Groupe Traité'])

plt.tight_layout()
plt.savefig("did_trends.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

did_trends_data = did_data.copy()

# 3. MODÉLISATION SIMPLE ET CLAIRE
# Modèle DiD pour les scores aux tests
formula_score = f"{col['score_tests']} ~ {col['interaction_did']} + {col['reforme']} + {col['post']} + {col['budget_education']} + {col['ratio_eleves_enseignant']} + {col['taux_pauvrete']} + {col['niveau_urbanisation']}"
model_score = smf.ols(formula_score, data=df).fit()
print("\nRésultats du modèle DiD pour les scores aux tests:")
print(model_score.summary())

# Modèle DiD pour le taux d'emploi des jeunes
formula_emploi = f"{col['taux_emploi_jeunes']} ~ {col['interaction_did']} + {col['reforme']} + {col['post']} + {col['budget_education']} + {col['ratio_eleves_enseignant']} + {col['taux_pauvrete']} + {col['niveau_urbanisation']}"
model_emploi = smf.ols(formula_emploi, data=df).fit()
print("\nRésultats du modèle DiD pour le taux d'emploi des jeunes:")
print(model_emploi.summary())

# 4. TESTS DE BASE
# Analyse des résidus pour le modèle des scores aux tests
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# QQ-plot
sm.qqplot(model_score.resid, line='s', ax=axes[0])
axes[0].set_title('QQ-plot des Résidus (Scores aux Tests)')
axes[0].set_xlabel('Quantiles théoriques')
axes[0].set_ylabel('Quantiles observés')

# Résidus vs valeurs prédites
sns.scatterplot(x=model_score.fittedvalues, y=model_score.resid, ax=axes[1])
axes[1].set_title('Résidus vs Valeurs Prédites (Scores aux Tests)')
axes[1].set_xlabel('Valeurs prédites')
axes[1].set_ylabel('Résidus')

plt.tight_layout()
plt.savefig("residuals_score.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Analyse des résidus pour le modèle du taux d'emploi des jeunes
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# QQ-plot
sm.qqplot(model_emploi.resid, line='s', ax=axes[0])
axes[0].set_title('QQ-plot des Résidus (Taux d\'Emploi des Jeunes)')
axes[0].set_xlabel('Quantiles théoriques')
axes[0].set_ylabel('Quantiles observés')

# Résidus vs valeurs prédites
sns.scatterplot(x=model_emploi.fittedvalues, y=model_emploi.resid, ax=axes[1])
axes[1].set_title('Résidus vs Valeurs Prédites (Taux d\'Emploi des Jeunes)')
axes[1].set_xlabel('Valeurs prédites')
axes[1].set_ylabel('Résidus')

plt.tight_layout()
plt.savefig("residuals_emploi.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Test de multicolinéarité (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Sélection des variables pour le test de multicolinéarité
vif_data = df[['budget_education', 'nb_eleves', 'ratio_eleves_enseignant', 'taux_pauvrete', 'niveau_urbanisation']]

# Calcul des VIF
vif = pd.DataFrame()
vif["Variable"] = vif_data.columns
vif["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

print("\nFacteurs d'Inflation de la Variance (VIF):")
print(vif)

print("\nAnalyse terminée.")