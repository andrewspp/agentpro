import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configuration des paramètres
n_types_etablissements = 5      # Nombre de types d'établissements (fixé à 5)
n_etablissements_par_type = 40  # Nombre d'établissements par type (40 × 5 = 200 établissements)
n_periodes = 4                  # 4 périodes (au lieu de 16 trimestres)
date_reforme = 2                # Période où la réforme est implémentée
prop_etablissements_reformes = 0.45  # Proportion d'établissements qui adoptent la réforme

# Types d'établissements (réduit à 5 types principaux)
types_etablissements = [
    "Lycée",
    "Collège",
    "Primaire",
    "Maternelle",
    "Centre Professionnel"
]

# Effet réel de la réforme (ce qu'on cherche à estimer avec DiD)
EFFET_REFORME_SCORE = 3.2  # Points supplémentaires aux tests standardisés
EFFET_REFORME_EMPLOI = 5.5  # Points de pourcentage d'augmentation du taux d'emploi

# Création d'un tableau d'établissements avec leurs types
etablissements = []
for type_idx, type_nom in enumerate(types_etablissements):
    for i in range(n_etablissements_par_type):
        etablissement_id = type_idx * n_etablissements_par_type + i + 1
        etablissements.append({
            "id": etablissement_id,
            "type": type_nom
        })

# Nombre total d'établissements
n_etablissements_total = len(etablissements)

# Assignation du statut de traitement (réforme) à chaque établissement
statut_reforme = np.random.choice([0, 1], size=n_etablissements_total, p=[1-prop_etablissements_reformes, prop_etablissements_reformes])

# Création du dataset de panel
data_rows = []

# Date de départ 
date_debut = datetime(2015, 1, 1)

# Types d'établissements (réduit à 5 types principaux)
types_etablissements = [
    "Lycée",
    "Collège",
    "Primaire",
    "Maternelle",
    "Centre Professionnel"
]

for periode in range(1, n_periodes + 1):
    # Déterminer si c'est une période post-réforme
    post_reforme = 1 if periode >= date_reforme else 0
    
    # Calculer la date réelle pour cette période (annualisée au lieu de trimestrielle)
    date_actuelle = date_debut + timedelta(days=365 * (periode - 1))
    date_str = date_actuelle.strftime("%Y-%m-%d")
    annee = date_actuelle.year
    semestre = 1 if date_actuelle.month <= 6 else 2
    
    for etab_idx, etablissement in enumerate(etablissements):
        # Récupérer les infos de l'établissement
        etablissement_id = etablissement["id"]
        type_etablissement = etablissement["type"]
        
        # Statut de réforme pour cet établissement
        etablissement_reforme = statut_reforme[etab_idx]
        
        # Variables de contexte éducatif
        budget_education = np.random.normal(1000, 200) + (periode * 10)  # Augmente légèrement avec le temps
        nb_eleves = np.random.normal(500, 150) + (periode * 5)  # Croissance des effectifs
        ratio_eleves_enseignant = np.random.normal(22, 5)  # Ratio élèves/enseignant
        taux_pauvrete = max(5, min(40, np.random.normal(20, 8)))  # Taux de pauvreté environnant (%)
        niveau_urbanisation = max(0, min(100, np.random.normal(60, 25)))  # % population urbaine
        
        # Choix d'une approche pédagogique de base
        approche_pedagogique = np.random.choice(["Traditionnelle", "Progressive", "Mixte", "Expérimentale"])
        
        # Tendances temporelles générales (affectent tous les établissements)
        tendance_score = periode * 0.8  # Amélioration générale des scores par période
        tendance_emploi = periode * 0.5  # Amélioration de l'emploi des jeunes par période
        
        # Effet saisonnier (simplifié par semestre)
        effet_saisonnier = {1: 0.3, 2: -0.3}.get(semestre, 0)
        
        # Effet fixe du groupe traité (différence préexistante)
        effet_groupe_score = etablissement_reforme * (-1.5)  # Établissements qui adopteront la réforme ont des scores initialement plus faibles
        effet_groupe_emploi = etablissement_reforme * (-2.0)  # Et des taux d'emploi des jeunes plus faibles
        
        # Effet temporel global (après réforme, affecte tout le monde)
        effet_temporel_score = post_reforme * 1.0
        effet_temporel_emploi = post_reforme * 0.8
        
        # L'effet DiD: uniquement pour les établissements réformés après l'intervention
        effet_reforme_score = etablissement_reforme * post_reforme * EFFET_REFORME_SCORE
        effet_reforme_emploi = etablissement_reforme * post_reforme * EFFET_REFORME_EMPLOI
        
        # Effets des variables de contrôle sur les scores
        effet_budget = 0.003 * budget_education
        effet_ratio = -0.15 * ratio_eleves_enseignant  # Ratio élevé = mauvais pour les scores
        effet_pauvrete = -0.10 * taux_pauvrete  # Pauvreté = mauvais pour les scores
        effet_urbanisation = 0.01 * niveau_urbanisation
        
        # Effets des approches pédagogiques
        effet_approche = {
            "Traditionnelle": 0,
            "Progressive": 1.2,
            "Mixte": 0.7,
            "Expérimentale": 2.0 if periode > n_periodes/2 else -1.0  # Effet qui change avec le temps
        }.get(approche_pedagogique, 0)
        
        # Effets des types d'établissements (différences structurelles)
        effet_type_score = {
            "Lycée": 2.0,
            "Collège": 0.5,
            "Primaire": -0.5,
            "Maternelle": -1.0,
            "Centre Professionnel": 1.0
        }.get(type_etablissement, 0)
        
        effet_type_emploi = {
            "Lycée": 2.0,
            "Collège": 0.0,
            "Primaire": -1.0,
            "Maternelle": -2.0,
            "Centre Professionnel": 5.0
        }.get(type_etablissement, 0)
        
        # Termes d'erreur aléatoires
        erreur_score = np.random.normal(0, 2)
        erreur_emploi = np.random.normal(0, 3)
        
        # Variables de résultat: combinent tous les effets
        score_tests = (75 + tendance_score + effet_saisonnier + effet_groupe_score + 
                     effet_temporel_score + effet_reforme_score + effet_budget + 
                     effet_ratio + effet_pauvrete + effet_urbanisation + 
                     effet_approche + effet_type_score + erreur_score)
        
        taux_emploi_jeunes = (50 + tendance_emploi + effet_groupe_emploi + 
                            effet_temporel_emploi + effet_reforme_emploi + 
                            (0.5 * effet_approche) + effet_type_emploi + erreur_emploi)
        
        # S'assurer que les valeurs sont dans des plages raisonnables
        score_tests = max(30, min(100, score_tests))
        taux_emploi_jeunes = max(10, min(90, taux_emploi_jeunes))
        
        # Ajouter la ligne au dataset
        data_rows.append({
            "etablissement_id": etablissement_id,
            "type_etablissement": type_etablissement,
            "periode": periode,
            "date": date_str,
            "annee": annee,
            "semestre": semestre,
            "reforme": etablissement_reforme,
            "post": post_reforme,
            "interaction_did": etablissement_reforme * post_reforme,
            "budget_education": budget_education,
            "nb_eleves": nb_eleves,
            "ratio_eleves_enseignant": ratio_eleves_enseignant,
            "taux_pauvrete": taux_pauvrete,
            "niveau_urbanisation": niveau_urbanisation,
            "approche_pedagogique": approche_pedagogique,
            "score_tests": score_tests,
            "taux_emploi_jeunes": taux_emploi_jeunes
        })

# Créer le DataFrame
df = pd.DataFrame(data_rows)

# Introduire quelques valeurs manquantes pour plus de réalisme
colonnes_avec_manquants = ['budget_education', 'ratio_eleves_enseignant', 'taux_pauvrete', 
                          'niveau_urbanisation', 'score_tests', 'taux_emploi_jeunes']
for col in colonnes_avec_manquants:
    df[col] = df[col].mask(np.random.random(len(df)) < 0.02)

# Créer des variables supplémentaires utiles pour l'analyse
df['log_budget'] = np.log(df['budget_education'])
df['log_nb_eleves'] = np.log(df['nb_eleves'])

# Pour faciliter les régressions et graphiques DiD classiques
df['groupe'] = df['reforme'].map({1: 'Réformé', 0: 'Non réformé'})
df['periode_relative'] = df['periode'] - date_reforme  # Périodes par rapport à l'intervention

# Créer une variable phase pour simplifier davantage (3 phases distinctes)
df['phase'] = 'Pre-réforme'
df.loc[df['periode'] == date_reforme, 'phase'] = 'Implémentation'
df.loc[df['periode'] > date_reforme, 'phase'] = 'Post-réforme'

# Enregistrer les données
df.to_csv("donnees3.csv", index=False)

print(f"✅ Fichier 'reforme_education_did.csv' généré - Analyse de la réforme éducative")
print(f"   {n_etablissements_total} établissements observés ({n_etablissements_par_type} de chaque type) sur {n_periodes} périodes")
print(f"   Total observations: {n_etablissements_total * n_periodes} lignes dans le dataset")
print(f"   Réforme implémentée à la période {date_reforme} (P{date_reforme}) dans {prop_etablissements_reformes*100:.0f}% des établissements")
print(f"   Effet causal réel de la réforme: +{EFFET_REFORME_SCORE} points aux tests standardisés")
print(f"   et +{EFFET_REFORME_EMPLOI} points de pourcentage au taux d'emploi des jeunes")

# Afficher les statistiques descriptives clés pour vérification
print("\nScores moyens par groupe et période:")
print(df.groupby(['groupe', 'post'])['score_tests'].agg(['count', 'mean', 'std']).round(2))

print("\nTaux d'emploi des jeunes par groupe et période:")
print(df.groupby(['groupe', 'post'])['taux_emploi_jeunes'].agg(['count', 'mean', 'std']).round(2))