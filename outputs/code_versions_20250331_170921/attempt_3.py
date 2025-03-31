import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration de Seaborn pour un style visuel attrayant
sns.set_theme(style="whitegrid")

# Définition du dictionnaire des noms de colonnes pour faciliter l'accès
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
# Chargement du fichier CSV dans un DataFrame pandas
try:
    df = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/donnees2.csv')
    print("Fichier CSV chargé avec succès.")

    # Affichage des premières lignes du DataFrame pour vérification
    print("\nPremières lignes du DataFrame:")
    print(df.head())

    # Statistiques descriptives initiales
    print("\nStatistiques descriptives initiales:")
    print(df.describe())

    # Gestion des valeurs manquantes (imputation par la médiane pour les variables numériques)
    for column in df.columns:
        if df[column].isnull().any():
            if pd.api.types.is_numeric_dtype(df[column]):
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
                print(f"Valeurs manquantes dans '{column}' imputées avec la médiane ({median_value}).")
            else:
                # Supprimer les lignes avec des valeurs manquantes dans les colonnes non numériques
                df.dropna(subset=[column], inplace=True)
                print(f"Lignes avec des valeurs manquantes dans '{column}' supprimées.")


    # Vérification des valeurs manquantes après imputation
    print("\nNombre de valeurs manquantes par colonne après imputation:")
    print(df.isnull().sum())

    # Gestion des outliers (exemple simple: suppression des valeurs hors de 3 écarts-types pour RevenuMensuel)
    mean_income = df[col["RevenuMensuel"]].mean()
    std_income = df[col["RevenuMensuel"]].std()
    df = df[(df[col["RevenuMensuel"]] >= mean_income - 3 * std_income) & (df[col["RevenuMensuel"]] <= mean_income + 3 * std_income)]
    print("\nOutliers gérés (valeurs hors de 3 écarts-types pour RevenuMensuel supprimées).")

    # 🔍 Sélection des colonnes numériques pour éviter les erreurs sur df_numeric.corr()
    df_numeric = df.select_dtypes(include='number')

except FileNotFoundError:
    print("Erreur: Le fichier CSV n'a pas été trouvé. Vérifiez le chemin d'accès.")
    exit()

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# 2.1 Matrice de corrélation
try:
    correlation_matrix = df_numeric.corr(numeric_only=True) # Only calculate correlations for numeric columns
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Matrice de Corrélation des Variables")
    plt.savefig("correlation_matrix.png")
    plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
    plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
    plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
    correlation_data = correlation_matrix # Stocke les données

except Exception as e:
    print(f"Erreur lors de la création de la matrice de corrélation: {e}")
    correlation_data = None


# 2.2 Distributions des variables principales
# Distribution du revenu mensuel
plt.figure(figsize=(10, 6))
sns.histplot(df[col["RevenuMensuel"]], kde=True)
plt.title("Distribution du Revenu Mensuel")
plt.xlabel("Revenu Mensuel")
plt.ylabel("Fréquence")
plt.savefig("distribution_revenu.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
distribution_revenu_data = df[col["RevenuMensuel"]].copy() # Stocke les données

# Distribution de l'âge
plt.figure(figsize=(10, 6))
sns.histplot(df[col["Age"]], kde=True)
plt.title("Distribution de l'Âge")
plt.xlabel("Âge")
plt.ylabel("Fréquence")
plt.savefig("distribution_age.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
distribution_age_data = df[col["Age"]].copy() # Stocke les données

# Distribution des années d'éducation
plt.figure(figsize=(10, 6))
sns.histplot(df[col["EducationAnnees"]], kde=True)
plt.title("Distribution des Années d'Éducation")
plt.xlabel("Années d'Éducation")
plt.ylabel("Fréquence")
plt.savefig("distribution_education.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
distribution_education_data = df[col["EducationAnnees"]].copy() # Stocke les données


# 2.3 Relation entre le revenu mensuel et les années d'éducation
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[col["EducationAnnees"]], y=df[col["RevenuMensuel"]])
plt.title("Relation entre le Revenu Mensuel et les Années d'Éducation")
plt.xlabel("Années d'Éducation")
plt.ylabel("Revenu Mensuel")
plt.savefig("relation_education_revenu.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
relation_education_revenu_data = df[[col["EducationAnnees"], col["RevenuMensuel"]]].copy() # Stocke les données

# 2.4 Diagramme de boîtes pour le revenu mensuel par sexe
plt.figure(figsize=(8, 6))
sns.boxplot(x=df[col["Sexe"]], y=df[col["RevenuMensuel"]])
plt.title("Revenu Mensuel par Sexe")
plt.xlabel("Sexe")
plt.ylabel("Revenu Mensuel")
plt.savefig("revenu_par_sexe.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
revenu_par_sexe_data = df[[col["Sexe"], col["RevenuMensuel"]]].copy()  # Stocke les données

# 2.5 Diagramme de boîtes pour le revenu mensuel par continent
plt.figure(figsize=(12, 6))
sns.boxplot(x=df[col["Continent"]], y=df[col["RevenuMensuel"]])
plt.title("Revenu Mensuel par Continent")
plt.xlabel("Continent")
plt.ylabel("Revenu Mensuel")
plt.xticks(rotation=45)
plt.tight_layout()  # Ajuste la disposition pour éviter le chevauchement des étiquettes
plt.savefig("revenu_par_continent.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
revenu_par_continent_data = df[[col["Continent"], col["RevenuMensuel"]]].copy() # Stocke les données

# 3. MODÉLISATION SIMPLE ET CLAIRE

# Conversion des variables catégorielles en variables indicatrices (dummy variables)
df = pd.get_dummies(df, columns=[col["Sexe"], col["Continent"], col["Travaille"]], drop_first=True)

# Renommage des colonnes générées par pd.get_dummies pour éviter les problèmes avec statsmodels
df.rename(columns={'Sexe_Homme': 'Sexe_Homme', 'Continent_Afrique': 'Continent_Afrique',
                    'Continent_AmeriqueDuNord': 'Continent_AmeriqueDuNord', 'Continent_Antarctique': 'Continent_Antarctique',
                    'Continent_Asie': 'Continent_Asie', 'Continent_Europe': 'Continent_Europe',
                    'Continent_Oceanie': 'Continent_Oceanie', 'Travaille_Oui': 'Travaille_Oui'}, inplace=True)

# Construction du modèle de régression linéaire multiple
# Utilisation de statsmodels pour obtenir des statistiques complètes
formula = f'{col["RevenuMensuel"]} ~ {col["EducationAnnees"]} + {col["Age"]} + np.square({col["Age"]}) + Sexe_Homme + {col["AccesInternet"]} + {col["TailleMenage"]} + Travaille_Oui + Continent_Afrique + Continent_AmeriqueDuNord + Continent_Antarctique + Continent_Asie + Continent_Europe + Continent_Oceanie'

model = smf.ols(formula=formula, data=df)
results = model.fit()

# Affichage des résultats de la régression
print("\nRésultats de la régression:")
print(results.summary())

# 4. TESTS DE BASE

# Analyse des résidus
plt.figure(figsize=(10, 6))
sns.residplot(x=results.fittedvalues, y=results.resid, lowess=True, line_kws={'color': 'red'})
plt.title("Analyse des Résidus")
plt.xlabel("Valeurs Prédites")
plt.ylabel("Résidus")
plt.savefig("analyse_residus.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
analyse_residus_data = pd.DataFrame({'fitted_values': results.fittedvalues, 'residuals': results.resid}) # Stocke les données

# Test de multicolinéarité (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = model.exog_names
vif_data["VIF"] = [variance_inflation_factor(model.exog, i) for i in range(model.exog.shape[1])]

print("\nFacteurs d'Inflation de la Variance (VIF):")
print(vif_data)