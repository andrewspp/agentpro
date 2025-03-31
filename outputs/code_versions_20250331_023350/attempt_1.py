import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration des styles seaborn
sns.set_theme(style="whitegrid")

# Dictionnaire des noms de colonnes
col = {
    "IndividuID": "IndividuID",
    "Continent": "Continent",
    "Age": "Age",
    "Sexe": "Sexe",
    "EducationAnnees": "EducationAnnees",
    "Travaille": "Travaille",
    "AccesInternet": "AccesInternet",
    "TailleMenage": "TailleMenage",
    "RevenuMensuel": "RevenuMensuel",
    "DepensesMensuelles": "DepensesMensuelles"
}

# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
# Chargement des données
file_path = '/Users/pierreandrews/Desktop/AgentPro/donnees2.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/AgentPro/donnees2.csv')
# 🔍 Sélection des colonnes numériques pour éviter les erreurs sur df_numeric.corr()
df_numeric = df.select_dtypes(include='number')


# Affichage des premières lignes pour vérification
print("Premières lignes du DataFrame:")
print(df.head())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(df.describe())

# Gestion des valeurs manquantes
# Calcul du pourcentage de valeurs manquantes par colonne
missing_percentage = df.isnull().sum() / len(df) * 100
print("\nPourcentage de valeurs manquantes par colonne:")
print(missing_percentage)

# Imputation des valeurs manquantes (exemple avec la moyenne pour les colonnes numériques)
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        df[column].fillna(df[column].mean(), inplace=True)

# Suppression des lignes avec des valeurs manquantes (si l'imputation n'est pas appropriée)
# df.dropna(inplace=True)

# Vérification des valeurs manquantes après imputation
print("\nNombre de valeurs manquantes après imputation:")
print(df.isnull().sum().sum())

# Gestion des outliers (exemple simple : suppression des valeurs hors de 3 écarts-types)
for column in df.columns:
    if df[column].dtype in ['float64', 'int64']:
        mean = df[column].mean()
        std = df[column].std()
        df = df[(df[column] >= mean - 3*std) & (df[column] <= mean + 3*std)]

# Statistiques descriptives après nettoyage
print("\nStatistiques descriptives après nettoyage:")
print(df.describe())

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES
# 2.1 Matrice de corrélation
# Calcul de la matrice de corrélation
correlation_matrix = df_numeric.corr(numeric_only = True)

# Création de la figure
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
plt.title("Matrice de Corrélation des Variables")

# Sauvegarde et affichage
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Stockage des données utilisées
correlation_data = correlation_matrix.copy()  # Stocker une copie pour l'interprétation

# 2.2 Distributions des variables principales
# Distribution de l'âge
plt.figure(figsize=(8, 6))
sns.histplot(df[col["Age"]], kde=True)
plt.title("Distribution de l'Âge")
plt.xlabel("Âge")
plt.ylabel("Fréquence")
plt.savefig("age_distribution.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
age_distribution_data = df[col["Age"]].copy() # Stocker une copie

# Distribution du revenu mensuel (transformée logarithmiquement pour une meilleure visualisation)
plt.figure(figsize=(8, 6))
sns.histplot(np.log1p(df[col["RevenuMensuel"]]), kde=True)
plt.title("Distribution du Revenu Mensuel (Log)")
plt.xlabel("Log(Revenu Mensuel)")
plt.ylabel("Fréquence")
plt.savefig("revenu_distribution_log.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
revenu_distribution_data = np.log1p(df[col["RevenuMensuel"]]).copy() # Stocker une copie

# 2.3 Relation entre l'éducation et le revenu
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[col["EducationAnnees"]], y=df[col["RevenuMensuel"]], alpha=0.5)
plt.title("Relation entre l'Éducation et le Revenu Mensuel")
plt.xlabel("Années d'Éducation")
plt.ylabel("Revenu Mensuel")
plt.savefig("education_revenu_scatter.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
education_revenu_data = df[[col["EducationAnnees"], col["RevenuMensuel"]]].copy() # Stocker une copie

# 2.4 Revenu mensuel par continent
plt.figure(figsize=(12, 6))
sns.boxplot(x=df[col["Continent"]], y=df[col["RevenuMensuel"]])
plt.title("Revenu Mensuel par Continent")
plt.xlabel("Continent")
plt.ylabel("Revenu Mensuel")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("revenu_continent_boxplot.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
revenu_continent_data = df[[col["Continent"], col["RevenuMensuel"]]].copy() # Stocker une copie

# 2.5 Relation entre l'âge et le revenu mensuel
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[col["Age"]], y=df[col["RevenuMensuel"]], alpha=0.5)
plt.title("Relation entre l'âge et le revenu mensuel")
plt.xlabel("Âge")
plt.ylabel("Revenu Mensuel")
plt.savefig("age_revenu_scatter.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
age_revenu_data = df[[col["Age"], col["RevenuMensuel"]]].copy() # Stocker une copie

# 3. MODÉLISATION SIMPLE ET CLAIRE
# Conversion des variables catégorielles en variables dummies
df = pd.get_dummies(df, columns=[col["Continent"], col["Sexe"], col["Travaille"]], drop_first=True)

# Définition du modèle
formula = f'{col["RevenuMensuel"]} ~ {col["Age"]} + {col["EducationAnnees"]} + {col["AccesInternet"]} + {col["TailleMenage"]} + C(Sexe_Homme) + C(Travaille_Oui)'

# Ajustement du modèle
model = smf.ols(formula=formula, data=df)
results = model.fit()

# Affichage des résultats
print(results.summary())

# 4. TESTS DE BASE
# Analyse des résidus
plt.figure(figsize=(8, 6))
sns.histplot(results.resid, kde=True)
plt.title("Distribution des Résidus")
plt.xlabel("Résidus")
plt.ylabel("Fréquence")
plt.savefig("residuals_distribution.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
residuals_data = results.resid.copy() # Stocker une copie

# Test de multicolinéarité (VIF)
vif_data = pd.DataFrame()
vif_data["feature"] = model.exog_names
vif_data["VIF"] = [variance_inflation_factor(model.exog, i) for i in range(len(model.exog_names))]

print(vif_data)

vif_data_saved = vif_data.copy() # Stocker une copie pour interprétation
