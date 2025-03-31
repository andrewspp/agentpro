import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration du style seaborn
sns.set_theme(style="whitegrid")

# Définition du dictionnaire des noms de colonnes
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

# Affichage des premières lignes du DataFrame
print("Premières lignes du DataFrame:")
print(df.head())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(df.describe())

# Gestion des valeurs manquantes
# Nombre de valeurs manquantes par colonne
print("\nNombre de valeurs manquantes par colonne:")
print(df.isnull().sum())

# Imputation des valeurs manquantes (utilisation de la médiane pour les variables numériques)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        df[column].fillna(df[column].median(), inplace=True)
    else:
        # Supprimer les lignes avec des valeurs manquantes dans les colonnes catégorielles
        df.dropna(subset=[column], inplace=True)

# Vérification des valeurs manquantes après imputation
print("\nNombre de valeurs manquantes après imputation:")
print(df.isnull().sum())

# Conversion des types de données si nécessaire
# Conversion de 'AccesInternet' en type entier
df[col["AccesInternet"]] = df[col["AccesInternet"]].astype(int)

# Statistiques descriptives après imputation
print("\nStatistiques descriptives après imputation:")
print(df.describe())

# Gestion des outliers (exemple simple : suppression des valeurs en dehors de 3 écarts-types)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        mean = df[column].mean()
        std = df[column].std()
        df = df[(df[column] >= mean - 3*std) & (df[column] <= mean + 3*std)]

# Statistiques descriptives après gestion des outliers
print("\nStatistiques descriptives après gestion des outliers:")
print(df.describe())

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# 2.1 Matrice de corrélation
correlation_data = df[[col["Age"], col["EducationAnnees"], col["RevenuMensuel"], col["DepensesMensuelles"], col["TailleMenage"]]]
corr_matrix = correlation_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de Corrélation des Variables Numériques")
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.2 Distribution du revenu mensuel
plt.figure(figsize=(10, 6))
sns.histplot(df[col["RevenuMensuel"]], kde=True)
plt.title("Distribution du Revenu Mensuel")
plt.xlabel("Revenu Mensuel")
plt.ylabel("Fréquence")
plt.savefig("income_distribution.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.3 Relation entre l'éducation et le revenu
education_income_data = df[[col["EducationAnnees"], col["RevenuMensuel"]]]

plt.figure(figsize=(10, 6))
sns.scatterplot(x=col["EducationAnnees"], y=col["RevenuMensuel"], data=df)
plt.title("Relation entre le Nombre d'Années d'Éducation et le Revenu Mensuel")
plt.xlabel("Nombre d'Années d'Éducation")
plt.ylabel("Revenu Mensuel")
plt.savefig("education_vs_income.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.4 Boîte à moustaches du revenu par continent
plt.figure(figsize=(12, 6))
sns.boxplot(x=col["Continent"], y=col["RevenuMensuel"], data=df)
plt.title("Distribution du Revenu Mensuel par Continent")
plt.xlabel("Continent")
plt.ylabel("Revenu Mensuel")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("income_by_continent.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 2.5 Relation entre l'âge et le revenu
age_income_data = df[[col["Age"], col["RevenuMensuel"]]]
plt.figure(figsize=(10, 6))
sns.scatterplot(x=col["Age"], y=col["RevenuMensuel"], data=df)
plt.title("Relation entre l'Âge et le Revenu Mensuel")
plt.xlabel("Âge")
plt.ylabel("Revenu Mensuel")
plt.savefig("age_vs_income.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 3. MODÉLISATION SIMPLE ET CLAIRE
# Création de variables indicatrices (dummies) pour les variables catégorielles
df = pd.get_dummies(df, columns=[col["Sexe"], col["Continent"], col["Travaille"]])

# Définition du modèle de régression
formula = f'{col["RevenuMensuel"]} ~ {col["EducationAnnees"]} + {col["Age"]} + I({col["Age"]}**2) + Sexe_Homme + TailleMenage + AccesInternet + Continent_Afrique + Continent_AmeriqueDuSud + Continent_Antarctique + Continent_Asie + Continent_Europe + Continent_Oceanie + Travaille_Oui'

# Ajustement du modèle
model = smf.ols(formula, data=df)
results = model.fit()

# Affichage des résultats
print("\nRésultats du modèle de régression:")
print(results.summary())

# 4. TESTS DE BASE
# 4.1 Analyse des résidus
plt.figure(figsize=(10, 6))
sns.residplot(x=results.fittedvalues, y=results.resid, lowess=True, line_kws={'color': 'red'})
plt.title("Analyse des Résidus")
plt.xlabel("Valeurs Ajustées")
plt.ylabel("Résidus")
plt.savefig("residual_analysis.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 4.2 Test de multicolinéarité (VIF)
def calculate_vif(data):
    vif = pd.DataFrame()
    vif["variables"] = data.columns
    vif["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif

# Sélection des variables pour le test de multicolinéarité
vif_data = df[[col["EducationAnnees"], col["Age"], "TailleMenage", "AccesInternet", "Sexe_Homme", "Continent_Afrique", "Continent_AmeriqueDuSud", "Continent_Antarctique", "Continent_Asie", "Continent_Europe", "Continent_Oceanie", "Travaille_Oui"]]

# Calcul des VIF
vif = calculate_vif(vif_data)
print("\nFacteurs d'inflation de la variance (VIF):")
print(vif)

# 5. CAPTURE ET STOCKAGE DES DONNÉES POUR INTERPRÉTATION
# Les DataFrames utilisés pour chaque visualisation sont déjà stockés dans des variables.
# - correlation_data: DataFrame utilisé pour la matrice de corrélation
# - age_income_data: DataFrame utilisé pour le graphique de relation entre l'âge et le revenu
# - education_income_data: DataFrame utilisé pour le graphique de relation entre l'éducation et le revenu
# - df : DataFrame utilisé pour le boxplot du revenu par continent et pour la distribution du revenu mensuel et pour la régression
# Pour accéder à ces données, vous pouvez simplement utiliser ces variables.

print("\nFin de l'analyse.")