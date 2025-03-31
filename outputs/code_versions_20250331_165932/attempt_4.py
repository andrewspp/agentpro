import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Configuration du style des graphiques
# plt.style.use('whitegrid') # Erreur: 'whitegrid' est un style seaborn, pas matplotlib
sns.set_style('whitegrid') # Correction: Utiliser sns.set_style pour les styles seaborn
sns.set_palette("viridis") # Utilisation d'une palette de couleurs attrayante


# Dictionnaire des noms de colonnes pour faciliter l'accès
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

##################################################
# 1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES     #
##################################################

# Chargement des données
file_path = '/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv'
# Correction: Assigner le DataFrame chargé à df. La ligne précédente était redondante.
df = pd.read_csv('/Users/pierreandrews/Desktop/agentpro/reforme_education_did.csv')


# Affichage des premières lignes pour vérification
print("Aperçu des données initiales:")
print(df.head())

# Statistiques descriptives initiales
print("\nStatistiques descriptives initiales:")
# Correction: Utiliser df.describe() sur le DataFrame complet pour inclure les types non numériques si nécessaire pour l'info générale
# ou spécifier include='number' si seules les stats numériques sont voulues.
print(df.describe(include='number')) # Afficher les statistiques uniquement pour les colonnes numériques

# Gestion des valeurs manquantes (imputation par la moyenne pour les colonnes numériques)
# Correction: Sélectionner explicitement les colonnes numériques pour l'imputation
numeric_cols_for_imputation = df.select_dtypes(include=['int64', 'float64']).columns
for column in numeric_cols_for_imputation:
    # Utiliser fillna sur la colonne spécifique et assigner le résultat ou utiliser inplace=True
    df[column].fillna(df[column].mean(), inplace=True)

# Vérification des valeurs manquantes après imputation
print("\nNombre de valeurs manquantes par colonne après imputation:")
print(df.isnull().sum())

# Gestion des outliers (exemple simple: suppression des valeurs hors de 3 écarts-types de la moyenne)
# Correction: Itérer sur les colonnes numériques pour éviter les erreurs sur les non-numériques
numeric_cols_for_outlier = df.select_dtypes(include=['int64', 'float64']).columns
# S'assurer que l'ID n'est pas traité comme une variable numérique ordinaire pour les outliers si c'est un identifiant
cols_to_check_outliers = [c for c in numeric_cols_for_outlier if c != col["etablissement_id"]]

print(f"\nTaille du DataFrame avant suppression des outliers: {df.shape}")
for column in cols_to_check_outliers:
    # Vérifier si la colonne existe toujours après les précédentes suppressions
    if column in df.columns:
        mean = df[column].mean()
        std = df[column].std()
        # Vérifier si l'écart-type est non nul pour éviter la division par zéro ou des conditions inutiles
        if std > 0:
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
            initial_rows = df.shape[0]
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            rows_removed = initial_rows - df.shape[0]
            if rows_removed > 0:
                print(f"Suppression de {rows_removed} outliers pour la colonne '{column}'")
        else:
             print(f"Écart-type nul pour la colonne '{column}', pas de suppression d'outliers basée sur l'écart-type.")


print(f"Taille du DataFrame après suppression des outliers: {df.shape}")

# Affichage des statistiques descriptives après nettoyage
print("\nStatistiques descriptives après nettoyage et imputation:")
print(df.describe(include='number')) # Afficher les statistiques uniquement pour les colonnes numériques

##################################################
# 2. VISUALISATIONS                               #
##################################################

# 2.1. Matrice de corrélation
# Correction: Sélectionner les colonnes numériques *après* le nettoyage et l'imputation
df_numeric_cleaned = df.select_dtypes(include='number')
corr_matrix = df_numeric_cleaned.corr()
plt.figure(figsize=(14, 12)) # Augmenter légèrement la taille pour une meilleure lisibilité
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5) # Ajouter des lignes pour séparer les cellules
plt.title("Matrice de Corrélation des Variables (Après Nettoyage)")
plt.tight_layout() # Ajuster pour éviter les chevauchements
plt.savefig("correlation_matrix.png")
# Correction: Supprimer les sauvegardes redondantes de 'temp_figure.png'
plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
correlation_data = corr_matrix

# 2.2. Distribution des scores aux tests
plt.figure(figsize=(8, 6))
# Vérifier si la colonne existe avant de tracer
if col["score_tests"] in df.columns:
    sns.histplot(df[col["score_tests"]], kde=True)
    plt.title("Distribution des Scores aux Tests")
    plt.xlabel("Score aux Tests")
    plt.ylabel("Fréquence")
    plt.tight_layout()
    plt.savefig("score_distribution.png")
    plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
    score_distribution_data = df[col["score_tests"]]
else:
    print(f"Colonne '{col['score_tests']}' non trouvée pour le graphique de distribution.")


# 2.3. Relation entre le budget d'éducation et les scores aux tests
plt.figure(figsize=(8, 6))
# Vérifier si les colonnes existent
if col["log_budget"] in df.columns and col["score_tests"] in df.columns:
    sns.scatterplot(x=df[col["log_budget"]], y=df[col["score_tests"]])
    plt.title("Relation entre Log du Budget d'Éducation et Scores aux Tests")
    plt.xlabel("Log du Budget d'Éducation")
    plt.ylabel("Score aux Tests")
    plt.tight_layout()
    plt.savefig("budget_vs_score.png")
    plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
    budget_score_data = df[[col["log_budget"], col["score_tests"]]]
else:
    print(f"Colonnes '{col['log_budget']}' ou '{col['score_tests']}' non trouvées pour le scatter plot.")


# 2.4. Évolution des scores aux tests avant et après la réforme par groupe
# Vérifier si les colonnes nécessaires existent
if all(c in df.columns for c in [col["periode"], col["groupe"], col["score_tests"]]):
    # Aggrégation des données
    # Correction: S'assurer que 'periode' et 'groupe' ne contiennent pas de NaN après le nettoyage si elles ne sont pas numériques
    # Si elles sont catégorielles, le groupby fonctionnera. Si elles étaient numériques et ont été nettoyées, c'est bon.
    did_data = df.groupby([col["periode"], col["groupe"]])[col["score_tests"]].mean().reset_index()

    # Création du graphique
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=col["periode"], y=col["score_tests"], hue=col["groupe"], data=did_data, marker='o') # Ajouter des marqueurs
    plt.title("Évolution des Scores aux Tests Avant et Après la Réforme")
    plt.xlabel("Période")
    plt.ylabel("Score aux Tests Moyen")
    # Correction: S'assurer que les ticks correspondent bien aux périodes uniques existantes
    if not did_data[col["periode"]].empty:
       plt.xticks(ticks=did_data[col["periode"]].unique(), labels=did_data[col["periode"]].unique())
    plt.legend(title="Groupe")
    plt.tight_layout()
    plt.savefig("did_plot.png")
    plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
    evolution_data = did_data
else:
    print(f"Colonnes nécessaires pour le graphique DiD non trouvées ({col['periode']}, {col['groupe']}, {col['score_tests']}).")


# 2.5. Boîte à moustaches du taux d'emploi des jeunes par type d'établissement
# Vérifier si les colonnes existent
if col["type_etablissement"] in df.columns and col["taux_emploi_jeunes"] in df.columns:
    plt.figure(figsize=(12, 7)) # Ajuster la taille si beaucoup de types
    sns.boxplot(x=col["type_etablissement"], y=col["taux_emploi_jeunes"], data=df)
    plt.title("Distribution du Taux d'Emploi des Jeunes par Type d'Établissement")
    plt.xlabel("Type d'Établissement")
    plt.ylabel("Taux d'Emploi des Jeunes")
    plt.xticks(rotation=45, ha='right') # Améliorer la lisibilité des labels x
    plt.tight_layout()
    plt.savefig("emploi_par_etablissement.png")
    plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
    emploi_etablissement_data = df[[col["type_etablissement"], col["taux_emploi_jeunes"]]]
else:
     print(f"Colonnes '{col['type_etablissement']}' ou '{col['taux_emploi_jeunes']}' non trouvées pour le box plot.")


##################################################
# 3. MODÉLISATION                               #
##################################################

# S'assurer que le DataFrame n'est pas vide après le nettoyage
if not df.empty:
    # 3.1. Préparation des variables
    # Vérifier si les colonnes existent avant de créer l'interaction
    if col["reforme"] in df.columns and col["post"] in df.columns:
        df['reforme_post'] = df[col["reforme"]] * df[col["post"]]  # Variable d'interaction DiD

        # Définir les variables de contrôle communes
        control_vars = [
            col['log_budget'],
            col['ratio_eleves_enseignant'],
            col['taux_pauvrete'],
            col['niveau_urbanisation']
        ]
        # Vérifier l'existence des variables de contrôle et de la variable d'interaction
        required_cols_model = [
            col['score_tests'], col['taux_emploi_jeunes'], col['reforme'], col['post'], 'reforme_post',
            col['etablissement_id'], col['periode']
        ] + control_vars

        missing_cols = [c for c in required_cols_model if c not in df.columns]

        if not missing_cols:
            # Construction de la partie contrôle de la formule
            controls_formula_part = " + ".join(control_vars)

            # 3.2. Modèle DiD de base pour les scores aux tests
            try:
                formula_scores = f"{col['score_tests']} ~ {col['reforme']} + {col['post']} + reforme_post + {controls_formula_part} + C({col['etablissement_id']}) + C({col['periode']})"
                # S'assurer qu'il y a suffisamment de données après le nettoyage
                if df.shape[0] > (len(df[col['etablissement_id']].unique()) + len(df[col['periode']].unique()) + 5): # Heuristique simple
                    model_scores = smf.ols(formula_scores, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[col["etablissement_id"]]})
                    print("\nRésultats du modèle DiD pour les scores aux tests:")
                    print(model_scores.summary())
                    scores_model_results = model_scores.summary()
                else:
                    print("\nPas assez de données pour estimer le modèle DiD pour les scores aux tests après nettoyage.")
                    model_scores = None # Définir à None si le modèle ne peut pas être ajusté
                    scores_model_results = "Modèle non estimé en raison de données insuffisantes."

            except Exception as e:
                print(f"\nErreur lors de l'ajustement du modèle pour les scores aux tests: {e}")
                model_scores = None
                scores_model_results = f"Erreur lors de l'estimation: {e}"

            # 3.3. Modèle DiD de base pour le taux d'emploi des jeunes
            try:
                formula_emploi = f"{col['taux_emploi_jeunes']} ~ {col['reforme']} + {col['post']} + reforme_post + {controls_formula_part} + C({col['etablissement_id']}) + C({col['periode']})"
                 # S'assurer qu'il y a suffisamment de données après le nettoyage
                if df.shape[0] > (len(df[col['etablissement_id']].unique()) + len(df[col['periode']].unique()) + 5): # Heuristique simple
                    model_emploi = smf.ols(formula_emploi, data=df).fit(cov_type='cluster', cov_kwds={'groups': df[col["etablissement_id"]]})
                    print("\nRésultats du modèle DiD pour le taux d'emploi des jeunes:")
                    print(model_emploi.summary())
                    emploi_model_results = model_emploi.summary()
                else:
                    print("\nPas assez de données pour estimer le modèle DiD pour le taux d'emploi après nettoyage.")
                    model_emploi = None # Définir à None si le modèle ne peut pas être ajusté
                    emploi_model_results = "Modèle non estimé en raison de données insuffisantes."

            except Exception as e:
                print(f"\nErreur lors de l'ajustement du modèle pour le taux d'emploi: {e}")
                model_emploi = None
                emploi_model_results = f"Erreur lors de l'estimation: {e}"

        else:
            print(f"\nModèles non exécutés car les colonnes suivantes sont manquantes: {missing_cols}")
            model_scores = None
            model_emploi = None
            scores_model_results = f"Colonnes manquantes: {missing_cols}"
            emploi_model_results = f"Colonnes manquantes: {missing_cols}"

    else:
        print(f"\nColonnes '{col['reforme']}' ou '{col['post']}' manquantes. Impossible de créer la variable d'interaction et d'exécuter les modèles.")
        model_scores = None
        model_emploi = None
        scores_model_results = "Variable d'interaction non créée."
        emploi_model_results = "Variable d'interaction non créée."

    ##################################################
    # 4. TESTS DE BASE                              #
    ##################################################

    # 4.1. Analyse des résidus (scores aux tests)
    if model_scores is not None and hasattr(model_scores, 'resid'):
        plt.figure(figsize=(8, 6))
        sns.residplot(x=model_scores.fittedvalues, y=model_scores.resid, lowess=True,
                      scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1}) # Améliorer la visibilité
        plt.title("Analyse des Résidus du Modèle (Scores aux Tests)")
        plt.xlabel("Valeurs Prédites")
        plt.ylabel("Résidus")
        plt.tight_layout()
        plt.savefig("residues_scores.png")
        plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
        scores_residues_data = pd.DataFrame({'fitted_values': model_scores.fittedvalues, 'residues': model_scores.resid})
    else:
        print("\nGraphique des résidus pour les scores non généré (modèle non ajusté ou erreur).")
        scores_residues_data = None

    # 4.2. Analyse des résidus (taux d'emploi des jeunes)
    if model_emploi is not None and hasattr(model_emploi, 'resid'):
        plt.figure(figsize=(8, 6))
        sns.residplot(x=model_emploi.fittedvalues, y=model_emploi.resid, lowess=True,
                      scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1}) # Améliorer la visibilité
        plt.title("Analyse des Résidus du Modèle (Taux d'Emploi des Jeunes)")
        plt.xlabel("Valeurs Prédites")
        plt.ylabel("Résidus")
        plt.tight_layout()
        plt.savefig("residues_emploi.png")
        plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')
plt.show()
        emploi_residues_data = pd.DataFrame({'fitted_values': model_emploi.fittedvalues, 'residues': model_emploi.resid})
    else:
        print("\nGraphique des résidus pour l'emploi non généré (modèle non ajusté ou erreur).")
        emploi_residues_data = None


    # 4.3. Test de multicolinéarité (VIF)
    # Sélectionner uniquement les variables de contrôle *numériques* existantes pour le VIF
    vif_cols_available = [c for c in control_vars if c in df.columns and df[c].dtype in ['int64', 'float64']]

    if vif_cols_available:
        # Ajouter une constante pour le calcul VIF, comme requis par statsmodels VIF
        # Utiliser une copie pour ne pas modifier df
        vif_data = df[vif_cols_available].copy()
        # Supprimer les lignes avec NaN potentiels *uniquement* pour le calcul VIF
        vif_data.dropna(inplace=True)

        # S'assurer qu'il y a assez de données après dropna
        if not vif_data.empty and vif_data.shape[0] > 1:
            # Ajouter la constante
            vif_data_with_const = sm.add_constant(vif_data, prepend=False) # Ajoute 'const' à la fin

            # Calculer VIF pour chaque variable (sauf la constante)
            try:
                vif = pd.DataFrame()
                vif["Variable"] = vif_data_with_const.columns[:-1] # Exclure 'const'
                vif["VIF"] = [variance_inflation_factor(vif_data_with_const.values, i) for i in range(vif_data_with_const.shape[1] - 1)]

                print("\nFacteurs d'Inflation de la Variance (VIF):")
                print(vif)
                vif_results = vif
            except Exception as e:
                 print(f"\nErreur lors du calcul du VIF : {e}")
                 vif_results = f"Erreur VIF: {e}"
        else:
            print("\nPas assez de données ou données constantes pour calculer le VIF après suppression des NaNs.")
            vif_results = "Données insuffisantes/constantes pour VIF."
    else:
        print("\nAucune variable de contrôle numérique disponible pour calculer le VIF.")
        vif_results = "Pas de variables pour VIF."

else:
    print("\nLe DataFrame est vide après le prétraitement. Arrêt de l'analyse.")
    # Assigner None aux variables de résultats pour éviter les erreurs si elles sont utilisées plus tard
    correlation_data = None
    score_distribution_data = None
    budget_score_data = None
    evolution_data = None
    emploi_etablissement_data = None
    scores_model_results = "DataFrame vide."
    emploi_model_results = "DataFrame vide."
    scores_residues_data = None
    emploi_residues_data = None
    vif_results = "DataFrame vide."

print("\n--- Fin du script ---")

