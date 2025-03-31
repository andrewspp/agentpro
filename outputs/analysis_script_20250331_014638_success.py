import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration du style Seaborn (IMPORTANT: utiliser 'whitegrid')
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
# Charger les données
try:
    df = pd.read_csv('/Users/pierreandrews/Desktop/AgentPro/donnees2.csv')
    print("Fichier CSV chargé avec succès.")
except FileNotFoundError:
    print("Erreur: Le fichier CSV n'a pas été trouvé. Vérifiez le chemin d'accès.")
    exit()

# Afficher les premières lignes pour vérification
print("Aperçu des premières lignes du DataFrame:")
print(df.head())

# Informations sur le DataFrame
print("\nInformations sur le DataFrame:")
print(df.info())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(df.describe())

# Gestion des valeurs manquantes (imputation par la médiane pour les numériques, mode pour les catégorielles)
for column in df.columns:
    if df[column].isnull().any():
        if pd.api.types.is_numeric_dtype(df[column]):
            median_val = df[column].median()
            df[column] = df[column].fillna(median_val)
            print(f"Valeurs manquantes dans '{column}' imputées avec la médiane: {median_val}")
        else:
            mode_val = df[column].mode()[0]
            df[column] = df[column].fillna(mode_val)
            print(f"Valeurs manquantes dans '{column}' imputées avec le mode: {mode_val}")

# Vérification des valeurs manquantes après imputation
print("\nNombre de valeurs manquantes après imputation:")
print(df.isnull().sum())


# Gestion des outliers (IQR method, simple removal)
for column in [col["Age"], col["EducationAnnees"], col["RevenuMensuel"], col["DepensesMensuelles"], col["TailleMenage"]]:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)] # Keep only rows *within* the bounds.

print("\nStatistiques descriptives après nettoyage (valeurs manquantes et outliers):")
print(df.describe())

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# Matrice de corrélation
correlation_matrix_data = df[[col["Age"], col["EducationAnnees"], col["RevenuMensuel"], col["DepensesMensuelles"], col["TailleMenage"], col["AccesInternet"]]].copy() # Dataframe sauvegardé
corr_matrix = correlation_matrix_data.corr(numeric_only=True)

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Matrice de Corrélation des Variables Numériques")
plt.tight_layout()  # Ajustement pour éviter que le titre ne soit coupé
plt.savefig('correlation_matrix.png') # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure

# Distribution de l'âge
age_distribution_data = df[[col["Age"]]].copy()
plt.figure(figsize=(10, 6))
ax = sns.histplot(age_distribution_data[col["Age"]], kde=True) # Ajout de KDE pour une meilleure visualisation
ax.set_xlabel("Âge", fontsize=12) # Ajout d'une description claire
ax.set_ylabel("Fréquence", fontsize=12) # Ajout d'une description claire
plt.title("Distribution de l'Âge", fontsize=14)
plt.tight_layout()
plt.savefig('age_distribution.png') # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure

# Distribution du revenu mensuel (transformée logarithmiquement)
df['LogRevenuMensuel'] = np.log1p(df[col["RevenuMensuel"]])  # Ajout de 1 pour éviter log(0)
revenu_distribution_data = df[['LogRevenuMensuel']].copy()
plt.figure(figsize=(10, 6))
ax = sns.histplot(revenu_distribution_data['LogRevenuMensuel'], kde=True) # Ajout de KDE pour une meilleure visualisation
ax.set_xlabel("Log(Revenu Mensuel)", fontsize=12) # Ajout d'une description claire
ax.set_ylabel("Fréquence", fontsize=12) # Ajout d'une description claire
plt.title("Distribution du Log du Revenu Mensuel", fontsize=14)
plt.tight_layout()
plt.savefig('revenu_distribution.png') # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure

# Relation entre l'éducation et le revenu mensuel
education_revenu_data = df[[col["EducationAnnees"], col["RevenuMensuel"]]].copy()
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x=education_revenu_data[col["EducationAnnees"]], y=education_revenu_data[col["RevenuMensuel"]])
ax.set_xlabel("Années d'Éducation", fontsize=12) # Ajout d'une description claire
ax.set_ylabel("Revenu Mensuel", fontsize=12) # Ajout d'une description claire
plt.title("Relation entre les Années d'Éducation et le Revenu Mensuel", fontsize=14)
plt.tight_layout()
plt.savefig('education_revenu_relation.png') # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure

# Revenu moyen par continent
continent_revenu_data = df[[col["Continent"], col["RevenuMensuel"]]].groupby(col["Continent"])[col["RevenuMensuel"]].mean().reset_index()
plt.figure(figsize=(12, 6))
ax = sns.barplot(x=continent_revenu_data[col["Continent"]], y=continent_revenu_data[col["RevenuMensuel"]], palette="viridis")
ax.set_xlabel("Continent", fontsize=12) # Ajout d'une description claire
ax.set_ylabel("Revenu Mensuel Moyen", fontsize=12) # Ajout d'une description claire
plt.title("Revenu Mensuel Moyen par Continent", fontsize=14)
plt.xticks(rotation=45, ha="right")  # Rotation des étiquettes pour la lisibilité
plt.tight_layout()
plt.savefig('continent_revenu.png') # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure

# 3. MODÉLISATION SIMPLE ET CLAIRE

# Encodage des variables catégorielles (Sexe, Travaille, Continent)
df = pd.get_dummies(df, columns=[col["Sexe"], col["Travaille"], col["Continent"]], drop_first=True)

# Préparation des variables pour la régression
X = df[[col["Age"], col["EducationAnnees"], 'Sexe_Homme', 'Travaille_Oui', 'Continent_Antarctique', 'Continent_Asie', 'Continent_Europe', 'Continent_Oceanie', 'Continent_AmeriqueDuNord', 'Continent_AmeriqueDuSud', col["AccesInternet"], col["TailleMenage"]]].astype(float)
y = df['LogRevenuMensuel'].astype(float)  # Utilisation du log du revenu

# Ajout d'une constante pour le terme d'interception
X = sm.add_constant(X)

# Construction du modèle de régression linéaire multiple
model = sm.OLS(y, X)

# Entraînement du modèle
results = model.fit()

# Affichage des résultats de la régression
print(results.summary())

# 4. TESTS DE BASE

# Analyse des résidus
residuals = results.resid
plt.figure(figsize=(10, 6))
sns.scatterplot(x=results.fittedvalues, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Valeurs Prédites")
plt.ylabel("Résidus")
plt.title("Analyse des Résidus")
plt.savefig('residuals_analysis.png') # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show() # Affichage de la figure

# Test de normalité des résidus (Jarque-Bera)
print("\nTest de Normalité des Résidus (Jarque-Bera):")
print(sm.stats.jarque_bera(residuals))

# Vérification de la multicolinéarité (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nFacteurs d'Inflation de la Variance (VIF):")
print(vif_data)

print("\nAnalyse Complète Terminée.")