import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration du style Seaborn (IMPORTANT: 'whitegrid' est obsolète)
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
# Charger les données
file_path = '/Users/pierreandrews/Desktop/AgentPro/donnees2.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/AgentPro/donnees2.csv')

# Afficher les premières lignes pour vérification
print("Premières lignes du DataFrame:")
print(df.head())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(df.describe())

# Gestion des valeurs manquantes
print("\nValeurs manquantes par colonne avant imputation:")
print(df.isnull().sum())

# Imputation des valeurs manquantes (utilisation de la médiane pour les variables numériques)
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        df[column] = df[column].fillna(df[column].median())

# Gestion des valeurs manquantes pour les variables catégorielles (utilisation de la valeur la plus fréquente)
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])

print("\nValeurs manquantes par colonne après imputation:")
print(df.isnull().sum())

# Conversion des variables catégorielles en numériques (variables binaires)
df[col["Travaille"]] = df[col["Travaille"]].map({'Oui': 1, 'Non': 0})
df[col["Sexe"]] = df[col["Sexe"]].map({'Homme': 1, 'Femme': 0})

# Création de variables dummies pour la variable Continent
df = pd.get_dummies(df, columns=[col["Continent"]], prefix='Continent', drop_first=True)

# Afficher les types de données après le prétraitement
print("\nTypes de données après le prétraitement:")
print(df.dtypes)

# Statistiques descriptives après le prétraitement
print("\nStatistiques descriptives après le prétraitement:")
print(df.describe())

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# 2.1 Matrice de corrélation
correlation_matrix_data = df[[col["RevenuMensuel"], col["EducationAnnees"], col["Travaille"], col["AccesInternet"], col["TailleMenage"], col["Age"], col["Sexe"]]]
corr_matrix = correlation_matrix_data.corr()
plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Matrice de Corrélation des Variables Principales")
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nFigure: correlation_matrix.png sauvegardée et affichée")

# 2.2 Distributions des variables principales
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Distribution de RevenuMensuel
revenu_mensuel_data = df[col["RevenuMensuel"]]
sns.histplot(revenu_mensuel_data, kde=True, ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title("Distribution du Revenu Mensuel")
axes[0, 0].set_xlabel("Revenu Mensuel")
axes[0, 0].set_ylabel("Fréquence")

# Distribution de EducationAnnees
education_annees_data = df[col["EducationAnnees"]]
sns.histplot(education_annees_data, kde=True, ax=axes[0, 1], color='lightgreen')
axes[0, 1].set_title("Distribution des Années d'Éducation")
axes[0, 1].set_xlabel("Années d'Éducation")
axes[0, 1].set_ylabel("Fréquence")

# Distribution de l'âge
age_data = df[col["Age"]]
sns.histplot(age_data, kde=True, ax=axes[1, 0], color='lightcoral')
axes[1, 0].set_title("Distribution de l'Âge")
axes[1, 0].set_xlabel("Âge")
axes[1, 0].set_ylabel("Fréquence")

# Distribution de TailleMenage
taille_menage_data = df[col["TailleMenage"]]
sns.histplot(taille_menage_data, kde=True, ax=axes[1, 1], color='lightyellow')
axes[1, 1].set_title("Distribution de la Taille du Ménage")
axes[1, 1].set_xlabel("Taille du Ménage")
axes[1, 1].set_ylabel("Fréquence")

plt.tight_layout()
plt.savefig("distributions_principales.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nFigure: distributions_principales.png sauvegardée et affichée")

# 2.3 Relation entre RevenuMensuel et EducationAnnees
revenu_education_data = df[[col["RevenuMensuel"], col["EducationAnnees"]]]
plt.figure(figsize=(10, 6))
sns.scatterplot(x=col["EducationAnnees"], y=col["RevenuMensuel"], data=revenu_education_data, alpha=0.6, color='purple')
plt.title("Relation entre le Revenu Mensuel et les Années d'Éducation")
plt.xlabel("Années d'Éducation")
plt.ylabel("Revenu Mensuel")
plt.savefig("revenu_vs_education.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nFigure: revenu_vs_education.png sauvegardée et affichée")

# 2.4 Boîtes à moustaches pour RevenuMensuel par Continent
revenu_continent_data = df[[col["RevenuMensuel"]] + [col for col in df.columns if 'Continent_' in col]]
# Créer une colonne Continent à partir des colonnes dummies
continent_cols = [col for col in df.columns if 'Continent_' in col]
df['Continent'] = df[continent_cols].idxmax(axis=1).str.replace('Continent_', '')

plt.figure(figsize=(14, 8))
sns.boxplot(x='Continent', y=col["RevenuMensuel"], data=df, palette='viridis')
plt.title("Distribution du Revenu Mensuel par Continent")
plt.xlabel("Continent")
plt.ylabel("Revenu Mensuel")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("revenu_par_continent.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nFigure: revenu_par_continent.png sauvegardée et affichée")

# 2.5 Revenu Mensuel par Travaille et AccesInternet
revenu_travaille_internet_data = df[[col["RevenuMensuel"], col["Travaille"], col["AccesInternet"]]]
plt.figure(figsize=(10, 6))
sns.barplot(x=col["Travaille"], y=col["RevenuMensuel"], hue=col["AccesInternet"], data=df, palette='muted')
plt.title("Revenu Mensuel en fonction de l'Emploi et de l'Accès à Internet")
plt.xlabel("Travaille (0=Non, 1=Oui)")
plt.ylabel("Revenu Mensuel Moyen")
plt.savefig("revenu_travail_internet.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nFigure: revenu_travail_internet.png sauvegardée et affichée")

# 3. MODÉLISATION SIMPLE ET CLAIRE

# 3.1 Modèle de base (OLS)
formula = f'{col["RevenuMensuel"]} ~ {col["EducationAnnees"]} + {col["Travaille"]} + {col["AccesInternet"]} + {col["TailleMenage"]} + {col["Age"]} + {col["Sexe"]} + ' + ' + '.join([col for col in df.columns if 'Continent_' in col])
model = smf.ols(formula=formula, data=df)
results = model.fit()
print("\nRésultats du modèle OLS de base:")
print(results.summary())

# 3.2 Modèle avec interaction
formula_interaction = f'{col["RevenuMensuel"]} ~ {col["EducationAnnees"]} + {col["Travaille"]} + {col["AccesInternet"]} + {col["TailleMenage"]} + {col["EducationAnnees"]}*{col["AccesInternet"]} + {col["Age"]} + {col["Sexe"]} + ' + ' + '.join([col for col in df.columns if 'Continent_' in col])
model_interaction = smf.ols(formula=formula_interaction, data=df)
results_interaction = model_interaction.fit()
print("\nRésultats du modèle OLS avec interaction:")
print(results_interaction.summary())

# 4. TESTS DE BASE

# 4.1 Test d'hétéroscédasticité (White)
white_test = het_white(results.resid, model.exog)
labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print('\nTest d\'hétéroscédasticité de White:')
for value, label in zip(white_test, labels):
    print(f'{label}: {value}')

# 4.2 Analyse des résidus (histogramme)
residuals_data = results.resid
plt.figure(figsize=(10, 6))
sns.histplot(residuals_data, kde=True)
plt.title("Distribution des Résidus")
plt.xlabel("Résidus")
plt.ylabel("Fréquence")
plt.savefig("residuals_distribution.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
print("\nFigure: residuals_distribution.png sauvegardée et affichée")

# 4.3 Vérification de la multicolinéarité (VIF)
# Sélection des variables indépendantes pour le calcul du VIF
independent_vars = df[[col["EducationAnnees"], col["Travaille"], col["AccesInternet"], col["TailleMenage"], col["Age"], col["Sexe"]] + [col for col in df.columns if 'Continent_' in col]]

# Convertir les colonnes catégorielles en numériques si nécessaire
for col_name in independent_vars.columns:
    if independent_vars[col_name].dtype == 'object':
        try:
            independent_vars[col_name] = pd.to_numeric(independent_vars[col_name])
        except ValueError:
            # Gérer les erreurs de conversion si nécessaire
            print(f"Impossible de convertir la colonne {col_name} en numérique.")

# Supprimer les colonnes avec des valeurs non numériques
independent_vars = independent_vars.select_dtypes(include=[np.number])

vif_data = pd.DataFrame()
vif_data["Variable"] = independent_vars.columns
vif_data["VIF"] = [variance_inflation_factor(independent_vars.values, i) for i in range(independent_vars.shape[1])]

print("\nFacteurs d'inflation de variance (VIF):")
print(vif_data)