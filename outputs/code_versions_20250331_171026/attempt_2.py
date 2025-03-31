import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Configuration du style des graphiques Seaborn
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
# Chargement du fichier CSV
try:
    data = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/donnees2.csv')
except FileNotFoundError:
    print("Fichier CSV non trouvé. Veuillez vérifier le chemin.")
    exit()

# Gestion des valeurs manquantes (imputation par la médiane pour les variables numériques)
for column in data.columns:
    if data[column].dtype in ['int64', 'float64']:
        median_value = data[column].median()
        data[column] = data[column].fillna(median_value)

# Gestion des valeurs manquantes pour les variables catégorielles (imputation par la valeur la plus fréquente)
for column in data.columns:
    if data[column].dtype == 'object':
        most_frequent_value = data[column].mode()[0]
        data[column] = data[column].fillna(most_frequent_value)


# Suppression des outliers (méthode simple basée sur l'écart interquartile - IQR)
for column in data.columns:
    if data[column].dtype in ['int64', 'float64']:
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


# Statistiques descriptives
print("\nStatistiques descriptives:\n", data.describe())

# 2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES

# 2.1 Matrice de corrélation
# Sélectionner uniquement les colonnes numériques pour la matrice de corrélation
numerical_data = data.select_dtypes(include=np.number)
corr_matrix = numerical_data.corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation des variables")
plt.savefig("correlation_matrix.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

corr_matrix_data = data.copy()  # Copie des données pour l'interprétation
print("Données utilisées pour la matrice de corrélation: corr_matrix_data")


# 2.2 Distribution du revenu mensuel
plt.figure(figsize=(10, 6))
sns.histplot(data[col["RevenuMensuel"]], kde=True)
plt.title("Distribution du revenu mensuel")
plt.xlabel("Revenu Mensuel")  # Ajout de l'étiquette de l'axe x
plt.ylabel("Fréquence")  # Ajout de l'étiquette de l'axe y
plt.savefig("revenu_mensuel_distribution.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

revenu_distribution_data = data.copy() #stocker le data frame utilisé pour la visualisation
print("Données utilisées pour la distribution du revenu mensuel: revenu_distribution_data")

# 2.3 Relation entre l'éducation et le revenu mensuel
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[col["EducationAnnees"]], y=data[col["RevenuMensuel"]])
plt.title("Relation entre les années d'éducation et le revenu mensuel")
plt.xlabel("Années d'éducation")  # Ajout de l'étiquette de l'axe x
plt.ylabel("Revenu Mensuel")  # Ajout de l'étiquette de l'axe y
plt.savefig("education_revenu_relation.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

education_revenu_data = data.copy() #stocker le data frame utilisé pour la visualisation
print("Données utilisées pour la relation entre éducation et revenu: education_revenu_data")

# 2.4 Boîtes à moustaches du revenu mensuel par continent
plt.figure(figsize=(12, 6))
sns.boxplot(x=data[col["Continent"]], y=data[col["RevenuMensuel"]])
plt.title("Revenu mensuel par continent")
plt.xlabel("Continent")  # Ajout de l'étiquette de l'axe x
plt.ylabel("Revenu Mensuel")  # Ajout de l'étiquette de l'axe y
plt.xticks(rotation=45)
plt.savefig("revenu_par_continent.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

continent_revenu_data = data.copy() #stocker le data frame utilisé pour la visualisation
print("Données utilisées pour le revenu par continent: continent_revenu_data")

# 2.5 Diagramme à barres du revenu mensuel moyen par sexe
plt.figure(figsize=(8, 6))
sns.barplot(x=data[col["Sexe"]], y=data[col["RevenuMensuel"]])
plt.title("Revenu mensuel moyen par sexe")
plt.xlabel("Sexe")  # Ajout de l'étiquette de l'axe x
plt.ylabel("Revenu Mensuel Moyen")  # Ajout de l'étiquette de l'axe y
plt.savefig("revenu_par_sexe.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

sexe_revenu_data = data.copy() #stocker le data frame utilisé pour la visualisation
print("Données utilisées pour le revenu par sexe: sexe_revenu_data")

# 3. MODÉLISATION SIMPLE ET CLAIRE
# Création du modèle de régression
model = smf.ols(f'{col["RevenuMensuel"]} ~ {col["EducationAnnees"]} + {col["Age"]} + C({col["Sexe"]}) + C({col["AccesInternet"]}) + {col["TailleMenage"]} + C({col["Continent"]}) + C({col["Travaille"]})', data=data)

# Ajustement du modèle
results = model.fit()

# Affichage des résultats
print(results.summary())

# 4. TESTS DE BASE
# Analyse des résidus
plt.figure(figsize=(10, 6))
sns.residplot(x=results.fittedvalues, y=results.resid, lowess=True)
plt.title("Analyse des résidus")
plt.xlabel("Valeurs prédites")
plt.ylabel("Résidus")
plt.savefig("residues_analysis.png")
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()

residues_analysis_data = data.copy() #stocker le data frame utilisé pour la visualisation
print("Données utilisées pour l'analyse des résidus: residues_analysis_data")

# Test d'hétéroscédasticité (Goldfeld-Quandt)
#from statsmodels.stats.diagnostic import het_goldfeldquandt
#gq_test = het_goldfeldquandt(results.resid, results.model.exog)
#print("Test de Goldfeld-Quandt: F statistic =", gq_test[0])
#print("Test de Goldfeld-Quandt: p-value =", gq_test[1])

# Vérification de la multicolinéarité (VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculer les VIF pour chaque variable indépendante
vif_data = pd.DataFrame()
vif_data["feature"] = model.exog_names
vif_data["VIF"] = [variance_inflation_factor(model.exog, i) for i in range(len(model.exog_names))]

print(vif_data)