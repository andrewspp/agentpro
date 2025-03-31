import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration du style seaborn
sns.set_theme(style='whitegrid')  # Utilisation du style whitegrid


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
# Chargement des données à partir du chemin absolu
file_path = '/Users/pierreandrews/Desktop/AgentPro/donnees2.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/AgentPro/donnees2.csv')

# Affichage des premières lignes du DataFrame pour vérification
print("Aperçu des données chargées:")
print(df.head())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(df.describe())

# Gestion des valeurs manquantes
# Affichage du nombre de valeurs manquantes par colonne
print("\nNombre de valeurs manquantes par colonne:")
print(df.isnull().sum())

# Imputation des valeurs manquantes (utilisation de la moyenne pour les colonnes numériques)
for column in df.columns:
    if df[column].dtype == 'float64':
        df[column].fillna(df[column].mean(), inplace=True)  # Remplissage par la moyenne
    elif df[column].dtype == 'object':
        df[column].fillna(df[column].mode()[0], inplace=True)  # Remplissage par le mode

# Vérification de l'absence de valeurs manquantes après imputation
print("\nVérification de l'absence de valeurs manquantes après imputation:")
print(df.isnull().sum())

# Gestion des outliers (exemple simple : suppression des valeurs hors des bornes raisonnables pour 'RevenuMensuel')
# Définition des bornes (à ajuster en fonction de la connaissance du domaine)
revenu_min = df[col["RevenuMensuel"]].quantile(0.01)
revenu_max = df[col["RevenuMensuel"]].quantile(0.99)

# Suppression des outliers
df = df[(df[col["RevenuMensuel"]] >= revenu_min) & (df[col["RevenuMensuel"]] <= revenu_max)]

# Statistiques descriptives après le prétraitement
print("\nStatistiques descriptives après le prétraitement:")
print(df.describe())

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# Matrice de corrélation
correlation_data = df[[col["Age"], col["EducationAnnees"], col["RevenuMensuel"], col["DepensesMensuelles"], col["TailleMenage"], col["AccesInternet"]]].copy()
correlation_matrix = correlation_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Matrice de Corrélation des Variables Numériques")
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Distributions des variables principales
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(df[col["Age"]], kde=True, ax=axes[0, 0], color='skyblue')
axes[0,0].set_xlabel("Age")
axes[0,0].set_ylabel("Fréquence")
sns.histplot(df[col["EducationAnnees"]], kde=True, ax=axes[0, 1], color='lightgreen')
axes[0,1].set_xlabel("Années d'éducation")
axes[0,1].set_ylabel("Fréquence")
sns.histplot(df[col["RevenuMensuel"]], kde=True, ax=axes[1, 0], color='salmon')
axes[1,0].set_xlabel("Revenu Mensuel")
axes[1,0].set_ylabel("Fréquence")
sns.histplot(df[col["DepensesMensuelles"]], kde=True, ax=axes[1, 1], color='gold')
axes[1,1].set_xlabel("Dépenses Mensuelles")
axes[1,1].set_ylabel("Fréquence")
fig.suptitle("Distributions des Variables Principales", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("distributions.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Relation entre le revenu mensuel et les années d'éducation
education_income_data = df[[col["EducationAnnees"], col["RevenuMensuel"]]].copy()
plt.figure(figsize=(10, 6))
sns.scatterplot(x=col["EducationAnnees"], y=col["RevenuMensuel"], data=df, alpha=0.5, color='purple')
plt.title("Relation entre le Revenu Mensuel et les Années d'Éducation")
plt.xlabel("Années d'Éducation")
plt.ylabel("Revenu Mensuel")
plt.savefig("education_vs_income.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Boîte à moustaches pour le revenu mensuel par continent
continent_income_data = df[[col["Continent"], col["RevenuMensuel"]]].copy()
plt.figure(figsize=(12, 6))
sns.boxplot(x=col["Continent"], y=col["RevenuMensuel"], data=df, palette='viridis')
plt.title("Revenu Mensuel par Continent")
plt.xlabel("Continent")
plt.ylabel("Revenu Mensuel")
plt.xticks(rotation=45)  # Rotation des étiquettes pour une meilleure lisibilité
plt.tight_layout()
plt.savefig("continent_vs_income.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# Barplot pour l'accès à Internet en fonction du sexe
gender_internet_data = df[[col["Sexe"], col["AccesInternet"]]].copy()
plt.figure(figsize=(8, 6))
sns.countplot(x=col["Sexe"], hue=col["AccesInternet"], data=df, palette='muted')
plt.title("Accès à Internet en Fonction du Sexe")
plt.xlabel("Sexe")
plt.ylabel("Nombre d'Individus")
plt.legend(title="Accès à Internet", labels=["Non", "Oui"])
plt.savefig("gender_vs_internet.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()


# 3. MODÉLISATION SIMPLE ET CLAIRE
# Conversion des variables catégorielles en variables dummy
df = pd.get_dummies(df, columns=[col["Continent"], col["Sexe"], col["Travaille"]], drop_first=True)

# Préparation des données pour le modèle
# Ajout d'une variable pour l'âge au carré
df['Age_squared'] = df[col["Age"]]**2

# Définition de la formule du modèle
formula = f'{col["RevenuMensuel"]} ~ {col["EducationAnnees"]} + C(Sexe_Homme) + {col["Age"]} + Age_squared + {col["AccesInternet"]} + {col["TailleMenage"]} + C(Continent_Afrique) + C(Continent_Antarctique) + C(Continent_Asie) + C(Continent_Europe) + C(Continent_Oceanie) + C(Continent_AmeriqueDuSud) + C(Travaille_Oui)'

# Estimation du modèle de régression linéaire multiple
model = smf.ols(formula=formula, data=df)
results = model.fit()

# Affichage des résultats du modèle
print("\nRésultats du modèle de régression:")
print(results.summary())

# 4. TESTS DE BASE
# Test de multicolinéarité (VIF)
vif_data = df[[col["EducationAnnees"], col["Age"], 'Age_squared', col["AccesInternet"], col["TailleMenage"]]]
vif = pd.DataFrame()
vif["Variable"] = vif_data.columns
vif["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

print("\nFacteur d'inflation de la variance (VIF):")
print(vif)

# Analyse des résidus (exemple simple : histogramme)
residuals = results.resid
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title("Histogramme des Résidus")
plt.xlabel("Résidus")
plt.ylabel("Fréquence")
plt.savefig("residuals_histogram.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

# 5. CAPTURE ET STOCKAGE DES DONNÉES POUR INTERPRÉTATION
# Les DataFrames utilisés pour chaque visualisation sont stockés dans des variables :
# - correlation_data
# - education_income_data
# - continent_income_data
# - gender_internet_data
# On conserve les dataframes initiaux et transformés pour toute interpretation future.
print("Script terminé.")