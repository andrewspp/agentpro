import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration du style des graphiques Seaborn
sns.set_theme(style="whitegrid")

# Dictionnaire des noms de colonnes pour faciliter l'accès
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
# Charger le fichier CSV en utilisant le chemin absolu
try:
    data = pd.read_csv('/Users/pierreandrews/Desktop/AgentPro/donnees2.csv')
    print("Fichier CSV chargé avec succès.")
except FileNotFoundError:
    print("Erreur: Le fichier CSV n'a pas été trouvé. Vérifiez le chemin d'accès.")
    exit()
except Exception as e:
    print(f"Erreur inattendue lors du chargement du fichier: {e}")
    exit()

# Afficher les premières lignes du DataFrame pour vérifier le chargement
print("\nPremières lignes du DataFrame:")
print(data.head())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
print(data.describe())

# Gestion des valeurs manquantes
print("\nNombre de valeurs manquantes par colonne avant imputation:")
print(data.isnull().sum())

# Imputation des valeurs manquantes (exemple : remplacement par la moyenne pour les colonnes numériques)
for column in data.columns:
    if data[column].dtype in ['int64', 'float64']:
        data[column] = data[column].fillna(data[column].mean())

# Vérification des valeurs manquantes après imputation
print("\nNombre de valeurs manquantes par colonne après imputation:")
print(data.isnull().sum())

# Gestion des outliers (exemple : suppression des outliers basés sur l'écart interquartile pour 'RevenuMensuel')
Q1 = data[col["RevenuMensuel"]].quantile(0.25)
Q3 = data[col["RevenuMensuel"]].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data[col["RevenuMensuel"]] >= lower_bound) & (data[col["RevenuMensuel"]] <= upper_bound)]

print("\nForme du DataFrame après le traitement des outliers:", data.shape)

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# 2.1 Matrice de corrélation
correlation_data = data[[col["Age"], col["EducationAnnees"], col["TailleMenage"], col["RevenuMensuel"], col["DepensesMensuelles"]]]
correlation_matrix = correlation_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
plt.title("Matrice de Corrélation")
plt.tight_layout()
plt.savefig("correlation_matrix.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure

# 2.2 Distribution de l'âge
age_data = data[col["Age"]]
plt.figure(figsize=(8, 6))
sns.histplot(age_data, kde=True, color="skyblue")
plt.title("Distribution de l'Âge")
plt.xlabel("Âge")  # Ajout de l'étiquette de l'axe x
plt.ylabel("Fréquence")  # Ajout de l'étiquette de l'axe y
plt.tight_layout()
plt.savefig("age_distribution.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure

# 2.3 Relation entre les années d'éducation et le revenu mensuel
education_income_data = data[[col["EducationAnnees"], col["RevenuMensuel"]]]
plt.figure(figsize=(10, 6))
sns.scatterplot(x=col["EducationAnnees"], y=col["RevenuMensuel"], data=education_income_data, color="coral")
plt.title("Relation entre les Années d'Éducation et le Revenu Mensuel")
plt.xlabel("Années d'Éducation")  # Ajout de l'étiquette de l'axe x
plt.ylabel("Revenu Mensuel")  # Ajout de l'étiquette de l'axe y
plt.tight_layout()
plt.savefig("education_income_relation.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure

# 2.4 Revenu mensuel par continent
continent_income_data = data[[col["Continent"], col["RevenuMensuel"]]]
plt.figure(figsize=(12, 6))
sns.boxplot(x=col["Continent"], y=col["RevenuMensuel"], data=continent_income_data, palette="viridis")
plt.title("Revenu Mensuel par Continent")
plt.xlabel("Continent")  # Ajout de l'étiquette de l'axe x
plt.ylabel("Revenu Mensuel")  # Ajout de l'étiquette de l'axe y
plt.xticks(rotation=45)  # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
plt.tight_layout()
plt.savefig("continent_income.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure

# 2.5 Revenu mensuel par sexe
gender_income_data = data[[col["Sexe"], col["RevenuMensuel"]]]
plt.figure(figsize=(8, 6))
sns.barplot(x=col["Sexe"], y=col["RevenuMensuel"], data=gender_income_data, palette="pastel")
plt.title("Revenu Mensuel par Sexe")
plt.xlabel("Sexe")  # Ajout de l'étiquette de l'axe x
plt.ylabel("Revenu Mensuel")  # Ajout de l'étiquette de l'axe y
plt.tight_layout()
plt.savefig("gender_income.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure

# 3. MODÉLISATION SIMPLE ET CLAIRE
# Préparation des données pour le modèle
data['Sexe'] = data['Sexe'].astype('category').cat.codes  # Conversion en numérique
data['AccesInternet'] = data['AccesInternet'].astype('category').cat.codes  # Conversion en numérique

# Création de variables indicatrices (dummies) pour la variable 'Continent'
data = pd.get_dummies(data, columns=['Continent'], drop_first=True)

# Définition de la formule du modèle
formula = f"{col['RevenuMensuel']} ~ {col['EducationAnnees']} + Sexe + {col['Age']} + {col['AccesInternet']} + {col['TailleMenage']}"

# Ajout des variables indicatrices pour le continent à la formule
continent_cols = [col for col in data.columns if 'Continent_' in col]
formula += " + " + " + ".join(continent_cols)

# Estimation du modèle
model = smf.ols(formula=formula, data=data)
results = model.fit()

# Affichage des résultats du modèle
print(results.summary())

# 4. TESTS DE BASE
# Analyse des résidus
plt.figure(figsize=(10, 6))
sns.regplot(x=results.fittedvalues, y=results.resid, lowess=True, line_kws={'color': 'red'})
plt.title("Analyse des Résidus")
plt.xlabel("Valeurs Prédites")
plt.ylabel("Résidus")
plt.tight_layout()
plt.savefig("residue_analysis.png")  # Sauvegarde de la figure
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()  # Affichage de la figure

# Test d'hétéroscédasticité (Goldfeld-Quandt)
gq_test = sm.stats.het_goldfeldquandt(results.resid, results.model.exog)
print("\nTest d'hétéroscédasticité (Goldfeld-Quandt):")
print("F-statistic:", gq_test[0])
print("p-value:", gq_test[1])

# Test de multicolinéarité (VIF)
def calculate_vif(dataframe):
    vif = pd.DataFrame()
    vif["variables"] = dataframe.columns
    vif["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
    return vif

# Sélection des variables indépendantes pour le calcul du VIF
independent_vars = data[[col['EducationAnnees'], 'Sexe', col['Age'], col['AccesInternet'], col['TailleMenage']] + continent_cols].astype(float)
vif = calculate_vif(independent_vars)
print("\nFacteurs d'Inflation de la Variance (VIF):")
print(vif)