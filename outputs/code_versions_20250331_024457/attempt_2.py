import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration du style Seaborn (IMPORTANT : 'whitegrid' est recommandé)
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
# Charger les données depuis le fichier CSV
file_path = '/Users/pierreandrews/Desktop/AgentPro/donnees2.csv'
df = pd.read_csv('/Users/pierreandrews/Desktop/AgentPro/donnees2.csv')
# 🔍 Sélection des colonnes numériques pour éviter les erreurs sur df_numeric.corr()
df_numeric = df.select_dtypes(include='number')


# Afficher les premières lignes du DataFrame pour vérification
print("Aperçu des données initiales:")
print(df.head())

# Informations générales sur le DataFrame
print("\nInformations sur le DataFrame:")
print(df.info())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(df.describe())

# Gestion des valeurs manquantes (imputation par la médiane pour les numériques)
for column in df.columns:
    if df[column].isnull().any() and pd.api.types.is_numeric_dtype(df[column]):
        median_val = df[column].median()
        df[column] = df[column].fillna(median_val)
        print(f"Valeurs manquantes dans '{column}' remplacées par la médiane: {median_val}")

# Gestion des valeurs manquantes (imputation par la valeur la plus fréquente pour les objets)
for column in df.columns:
    if df[column].isnull().any() and pd.api.types.is_object_dtype(df[column]):
        mode_val = df[column].mode()[0]
        df[column] = df[column].fillna(mode_val)
        print(f"Valeurs manquantes dans '{column}' remplacées par le mode: {mode_val}")

# Vérification des valeurs manquantes après imputation
print("\nNombre de valeurs manquantes après imputation:")
print(df.isnull().sum())

# Gestion des outliers (IQR method, seulement pour les variables revenu et dépenses)
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

df = remove_outliers_iqr(df, col["RevenuMensuel"])
df = remove_outliers_iqr(df, col["DepensesMensuelles"])

print("\nTaille du DataFrame après suppression des outliers:", df.shape)

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# 2.1 Matrice de corrélation
correlation_matrix = df_numeric.corr(numeric_only=True) #ajout de numeric_only car deprecated
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Matrice de Corrélation des Variables")
plt.savefig("correlation_matrix.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure
correlation_data = correlation_matrix #Stockage des données

# 2.2 Distributions des variables principales (RevenuMensuel, Age, EducationAnnees)
# RevenuMensuel
plt.figure(figsize=(10, 6))
sns.histplot(df[col["RevenuMensuel"]], kde=True)
plt.title("Distribution du Revenu Mensuel")
plt.xlabel("Revenu Mensuel")
plt.ylabel("Fréquence")
plt.savefig("revenu_mensuel_distribution.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure
revenu_data = df[col["RevenuMensuel"]] #Stockage des données

# Age
plt.figure(figsize=(10, 6))
sns.histplot(df[col["Age"]], kde=True)
plt.title("Distribution de l'Âge")
plt.xlabel("Âge")
plt.ylabel("Fréquence")
plt.savefig("age_distribution.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure
age_data = df[col["Age"]] #Stockage des données

# EducationAnnees
plt.figure(figsize=(10, 6))
sns.histplot(df[col["EducationAnnees"]], kde=True)
plt.title("Distribution des Années d'Éducation")
plt.xlabel("Années d'Éducation")
plt.ylabel("Fréquence")
plt.savefig("education_annees_distribution.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure
education_data = df[col["EducationAnnees"]] #Stockage des données

# 2.3 Relation entre RevenuMensuel et EducationAnnees
plt.figure(figsize=(10, 6))
sns.scatterplot(x=col["EducationAnnees"], y=col["RevenuMensuel"], data=df)
plt.title("Relation entre Revenu Mensuel et Années d'Éducation")
plt.xlabel("Années d'Éducation")
plt.ylabel("Revenu Mensuel")
plt.savefig("revenu_vs_education.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure
revenu_education_data = df[[col["EducationAnnees"], col["RevenuMensuel"]]] #Stockage des données

# 2.4 RevenuMensuel par Continent (Boxplot)
plt.figure(figsize=(12, 8))
sns.boxplot(x=col["Continent"], y=col["RevenuMensuel"], data=df)
plt.title("Revenu Mensuel par Continent")
plt.xlabel("Continent")
plt.ylabel("Revenu Mensuel")
plt.savefig("revenu_par_continent.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure
revenu_continent_data = df[[col["Continent"], col["RevenuMensuel"]]] #Stockage des données

# 2.5 Relation entre AccesInternet et RevenuMensuel
plt.figure(figsize=(10, 6))
sns.boxplot(x=col["AccesInternet"], y=col["RevenuMensuel"], data=df)
plt.title("Relation entre Accès Internet et Revenu Mensuel")
plt.xlabel("Accès Internet (0: Non, 1: Oui)")
plt.ylabel("Revenu Mensuel")
plt.savefig("acces_internet_vs_revenu.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure
acces_internet_revenu_data = df[[col["AccesInternet"], col["RevenuMensuel"]]] #Stockage des données

# 3. MODÉLISATION SIMPLE ET CLAIRE
# Préparation des données pour le modèle
df.loc[:, 'Sexe'] = df['Sexe'].map({'Homme': 1, 'Femme': 0}).astype(int)  # Conversion de 'Sexe' en numérique
df.loc[:, 'Travaille'] = df['Travaille'].map({'Oui': 1, 'Non': 0}).astype(int)  # Conversion de 'Travaille' en numérique
df = pd.get_dummies(df, columns=['Continent'], drop_first=True)  # Création de variables indicatrices pour les continents

# Définition des variables indépendantes et dépendante
X = df[[col["EducationAnnees"], "Sexe", col["AccesInternet"], col["TailleMenage"], "Travaille", col["Age"]]].copy()
X['Age_squared'] = X[col["Age"]]**2  # Ajout de l'âge au carré
for continent in df.columns:
    if "Continent_" in continent:
        X[continent] = df[continent]

X = sm.add_constant(X)  # Ajout de la constante
y = df[col["RevenuMensuel"]]

# Construction du modèle de régression linéaire multiple
model = sm.OLS(y, X.astype(float))

# Ajustement du modèle
results = model.fit()

# Affichage des résultats du modèle
print(results.summary())

# 4. TESTS DE BASE
# 4.1 Analyse des résidus
residuals = results.resid
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title("Distribution des Résidus")
plt.xlabel("Résidus")
plt.ylabel("Fréquence")
plt.savefig("residuals_distribution.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure
residuals_data = residuals #Stockage des données

# 4.2 Test d'hétéroscédasticité (Breusch-Pagan)
import statsmodels.stats.api as sms
names = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(residuals, model.exog)
print('\nTest d\'hétéroscédasticité (Breusch-Pagan):')
print(list(zip(names, test)))

# 4.3 Vérification de la multicolinéarité (VIF)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nFacteurs d'inflation de la variance (VIF):")
print(vif_data)