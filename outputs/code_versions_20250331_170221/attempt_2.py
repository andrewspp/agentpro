import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Configuration du style des graphiques seaborn
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
# Charger les données depuis le fichier CSV
file_path = '/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv')

# Afficher les premières lignes du DataFrame pour vérification
print("Aperçu du DataFrame:")
print(df.head())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(df.describe())

# Gestion des valeurs manquantes
print("\nNombre de valeurs manquantes par colonne avant imputation:")
print(df.isnull().sum())

# Imputation des valeurs manquantes (exemple avec la moyenne pour les colonnes numériques)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        df[column].fillna(df[column].mean(), inplace=True)

print("\nNombre de valeurs manquantes par colonne après imputation:")
print(df.isnull().sum())

# Conversion des variables catégorielles en dummies (One-Hot Encoding)
df = pd.get_dummies(df, columns=[col["type_etablissement"], col["approche_pedagogique"]])

# Afficher les premières lignes du DataFrame après le prétraitement
print("\nAperçu du DataFrame après le prétraitement:")
print(df.head())

# Statistiques descriptives après le prétraitement
print("\nStatistiques descriptives après le prétraitement:")
print(df.describe())

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# 2.1. Matrice de Corrélation
# Calculer la matrice de corrélation
corr_matrix = df[[col["score_tests"], col["taux_emploi_jeunes"], col["log_budget"], col["log_nb_eleves"], col["ratio_eleves_enseignant"], col["taux_pauvrete"], col["niveau_urbanisation"]]].corr()

# Créer une figure et un axe
fig, ax = plt.subplots(figsize=(12, 10))

# Générer et afficher la heatmap
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=.5, ax=ax)

# Ajouter un titre clair
ax.set_title("Matrice de Corrélation des Variables Clés")

# Ajuster les étiquettes
ax.set_xlabel("Variables")
ax.set_ylabel("Variables")

# Sauvegarder la figure
plt.savefig("correlation_matrix.png")

# Afficher la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Stockage des données utilisées pour la visualisation
correlation_data = corr_matrix

# 2.2 Distributions des Variables Principales
# Créer une figure avec deux sous-graphiques
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Distribution de 'score_tests'
sns.histplot(df[col["score_tests"]], kde=True, ax=axes[0], color="skyblue")
axes[0].set_title("Distribution des Scores aux Tests")
axes[0].set_xlabel("Score aux Tests")
axes[0].set_ylabel("Fréquence")

# Distribution de 'taux_emploi_jeunes'
sns.histplot(df[col["taux_emploi_jeunes"]], kde=True, ax=axes[1], color="lightcoral")
axes[1].set_title("Distribution du Taux d'Emploi des Jeunes")
axes[1].set_xlabel("Taux d'Emploi des Jeunes")
axes[1].set_ylabel("Fréquence")

# Ajuster la mise en page
plt.tight_layout()

# Sauvegarder la figure
plt.savefig("distributions_principales.png")

# Afficher la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Stockage des données utilisées pour la visualisation
distribution_score_tests_data = df[col["score_tests"]]
distribution_taux_emploi_jeunes_data = df[col["taux_emploi_jeunes"]]

# 2.3 Relation entre 'score_tests' et 'taux_emploi_jeunes'
# Créer un nuage de points
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=df[col["score_tests"]], y=df[col["taux_emploi_jeunes"]], ax=ax, color="mediumseagreen")
ax.set_title("Relation entre Score aux Tests et Taux d'Emploi des Jeunes")
ax.set_xlabel("Score aux Tests")
ax.set_ylabel("Taux d'Emploi des Jeunes")

# Sauvegarder la figure
plt.savefig("relation_score_emploi.png")

# Afficher la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Stockage des données utilisées pour la visualisation
relation_score_emploi_data = df[[col["score_tests"], col["taux_emploi_jeunes"]]]

# 2.4. Graphique de Tendance Temporelle avec DiD
# S'assurer que les données sont triées par 'etablissement_id' et 'periode'
df = df.sort_values(by=[col["etablissement_id"], col["periode"]])

# Créer un graphique de tendance temporelle pour les groupes traité et contrôle
fig, ax = plt.subplots(figsize=(14, 7))

# Calculer la moyenne des scores aux tests par période pour chaque groupe
df_temp = df.groupby([col["periode"], col["reforme"]])[col["score_tests"]].mean().reset_index()

# Tracer les tendances pour le groupe traité (reforme=1) et le groupe contrôle (reforme=0)
sns.lineplot(data=df_temp, x=col["periode"], y=col["score_tests"], hue=col["reforme"], marker='o', ax=ax)

# Ajouter une ligne verticale pour indiquer le moment de la réforme (période 2.5 car au milieu des périodes)
ax.axvline(x=2.5, color='red', linestyle='--', label='Réforme')

# Ajouter des étiquettes et un titre
ax.set_title("Tendances Temporelles des Scores aux Tests pour les Groupes Traité et Contrôle")
ax.set_xlabel("Période")
ax.set_ylabel("Score Moyen aux Tests")
ax.legend(title="Groupe", labels=["Contrôle (Non Réformé)", "Traité (Réformé)"])

# Sauvegarder la figure
plt.savefig("tendances_temporelles_did.png")

# Afficher la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Stockage des données utilisées pour la visualisation
tendances_temporelles_data = df.groupby([col["periode"], col["reforme"]])[col["score_tests"]].mean().reset_index()

# 3. MODÉLISATION SIMPLE ET CLAIRE
# 3.1 Préparation des variables pour le modèle
# Créer la variable d'interaction DiD
df['did'] = df[col["reforme"]] * df[col["post"]]

# Définition des noms des colonnes dummies
type_etablissement_dummies = [colname for colname in df.columns if col["type_etablissement"] in colname]
approche_pedagogique_dummies = [colname for colname in df.columns if col["approche_pedagogique"] in colname]

# Construction de la formule pour le modèle
formula_score = f'{col["score_tests"]} ~ did + {col["log_budget"]} + {col["log_nb_eleves"]} + {col["ratio_eleves_enseignant"]} + {col["taux_pauvrete"]} + {col["niveau_urbanisation"]} + C({col["annee"]})'
for dummy in type_etablissement_dummies:
    formula_score += f' + C({dummy})'
for dummy in approche_pedagogique_dummies:
    formula_score += f' + C({dummy})'

# Estimation du modèle DiD pour les scores aux tests
model_score = smf.ols(formula_score, data=df).fit()

# Affichage des résultats
print("\nRésultats du modèle DiD pour l'impact sur les scores aux tests:")
print(model_score.summary())

# 3.2 Formule du modèle DiD pour l'impact sur le taux d'emploi des jeunes
formula_emploi = f'{col["taux_emploi_jeunes"]} ~ did + {col["log_budget"]} + {col["log_nb_eleves"]} + {col["ratio_eleves_enseignant"]} + {col["taux_pauvrete"]} + {col["niveau_urbanisation"]} + C({col["annee"]})'
for dummy in type_etablissement_dummies:
    formula_emploi += f' + C({dummy})'
for dummy in approche_pedagogique_dummies:
    formula_emploi += f' + C({dummy})'

# Estimation du modèle DiD pour le taux d'emploi des jeunes
model_emploi = smf.ols(formula_emploi, data=df).fit()

# Affichage des résultats
print("\nRésultats du modèle DiD pour l'impact sur le taux d'emploi des jeunes:")
print(model_emploi.summary())

# Stockage des résultats du modèle pour interprétation
model_score_results = model_score.summary()
model_emploi_results = model_emploi.summary()

# 4. TESTS DE BASE
# 4.1 Analyse des résidus (Scores aux tests)
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(model_score.resid, kde=True, ax=ax)
ax.set_title("Distribution des Résidus (Scores aux Tests)")
ax.set_xlabel("Résidus")
ax.set_ylabel("Fréquence")
plt.savefig("residus_scores.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Stockage des données utilisées pour la visualisation
residus_scores_data = model_score.resid

# 4.2 Analyse des résidus (Taux d'emploi des jeunes)
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(model_emploi.resid, kde=True, ax=ax)
ax.set_title("Distribution des Résidus (Taux d'Emploi des Jeunes)")
ax.set_xlabel("Résidus")
ax.set_ylabel("Fréquence")
plt.savefig("residus_emploi.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Stockage des données utilisées pour la visualisation
residus_emploi_data = model_emploi.resid

# 4.3 Vérification de la multicolinéarité (Exemple avec VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Préparation des variables indépendantes pour le calcul du VIF
X = df[[col["log_budget"], col["log_nb_eleves"], col["ratio_eleves_enseignant"], col["taux_pauvrete"], col["niveau_urbanisation"]]]

# Calcul du VIF pour chaque variable
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nFacteurs d'Inflation de la Variance (VIF):")
print(vif_data)

# Stockage des données VIF pour interprétation
vif_results = vif_data

# 5. TEST DES TENDANCES PARALLÈLES
# Créer des termes d'interaction entre 'reforme' et 'periode_relative' pour les périodes pré-traitement
df['reforme_periode_relative_1'] = df[col['reforme']] * (df[col['periode_relative']] == -1)
df['reforme_periode_relative_0'] = df[col['reforme']] * (df[col['periode_relative']] == 0)
df['reforme_periode_relative_1_post'] = df[col['reforme']] * (df[col['periode_relative']] == 1)
df['reforme_periode_relative_2_post'] = df[col['reforme']] * (df[col['periode_relative']] == 2)

# Inclure ces termes d'interaction dans le modèle de régression
formula_tendances = f"{col['score_tests']} ~ {df.filter(like='reforme_periode_relative').columns.to_list()} + did + {col['log_budget']} + {col['log_nb_eleves']} + {col['ratio_eleves_enseignant']} + {col['taux_pauvrete']} + {col['niveau_urbanisation']} + C({col['annee']})"

# Ajout des variables catégorielles au modèle
for dummy in type_etablissement_dummies:
    formula_tendances += f' + C({dummy})'
for dummy in approche_pedagogique_dummies:
    formula_tendances += f' + C({dummy})'

# Estimer le modèle
model_tendances = smf.ols(formula_tendances, data=df).fit()

# Afficher les résultats du modèle
print("\nRésultats du test des tendances parallèles:")
print(model_tendances.summary())