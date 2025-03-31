import pandas as pd
import random
import numpy as np

# Liste des continents sans accent ni espace
continents = [
    "Afrique", "AmeriqueDuNord", "AmeriqueDuSud", "Antarctique",
    "Asie", "Europe", "Oceanie"
]

# Fonction pour ajouter des valeurs manquantes
def rand_val(base, variance, mini=None, maxi=None, dec=1, missing_prob=0.1):
    if random.random() < missing_prob:
        return np.nan
    val = round(random.uniform(base - variance, base + variance), dec)
    if mini is not None:
        val = max(val, mini)
    if maxi is not None:
        val = min(val, maxi)
    return val

# Nombre d'individus à générer
n = 10000

# Générer toutes les variables indépendantes
donnees_micro = {
    "IndividuID": list(range(1, n+1)),
    "Continent": [random.choice(continents) for _ in range(n)],
    "Age": [rand_val(35, 12, 18, 80, 0, missing_prob=0.05) for _ in range(n)],
    "Sexe": [random.choice(["Homme", "Femme"]) for _ in range(n)],
    "EducationAnnees": [rand_val(10, 4, 0, 20, 0, missing_prob=0.05) for _ in range(n)],
    "Travaille": [random.choice(["Oui", "Non"]) if random.random() < 0.95 else None for _ in range(n)],
    "AccesInternet": [random.choice([1, 0]) if random.random() > 0.1 else None for _ in range(n)],
    "TailleMenage": [rand_val(4, 2, 1, 12, 0, missing_prob=0.05) for _ in range(n)],
}

# Convertir en DataFrame pour faciliter le calcul du revenu
df_micro = pd.DataFrame(donnees_micro)

# Calculer le revenu mensuel basé sur les caractéristiques
revenus = []

for i in range(n):
    # Commencer avec un revenu de base
    revenu = 500
    
    # Effet de l'âge (linéaire et quadratique pour simuler une courbe en cloche)
    age = df_micro.loc[i, 'Age']
    if not pd.isna(age):
        revenu += 30 * age - 0.3 * (age ** 2)  # Augmente avec l'âge mais plafonne
    
    # Effet de l'éducation (fort impact positif)
    education = df_micro.loc[i, 'EducationAnnees']
    if not pd.isna(education):
        revenu += 200 * education
    
    # Effet du statut d'emploi (impact majeur)
    if df_micro.loc[i, 'Travaille'] == 'Oui':
        revenu += 800
    
    # Effet de l'accès à Internet (impact modéré)
    acces_internet = df_micro.loc[i, 'AccesInternet']
    if not pd.isna(acces_internet) and acces_internet == 1:
        revenu += 300
    
    # Effet de la taille du ménage (impact négatif)
    taille_menage = df_micro.loc[i, 'TailleMenage']
    if not pd.isna(taille_menage):
        revenu -= 50 * taille_menage
    
    # Effet du sexe (écart salarial)
    if df_micro.loc[i, 'Sexe'] == 'Homme':
        revenu += 200
    
    # Effet du continent (différences économiques régionales)
    continent = df_micro.loc[i, 'Continent']
    if continent == 'AmeriqueDuNord':
        revenu += 1000
    elif continent == 'AmeriqueDuSud':
        revenu += 500
    elif continent == 'Antarctique':
        revenu += 1500  # Salaires élevés pour les chercheurs
    elif continent == 'Asie':
        revenu += 700
    elif continent == 'Europe':
        revenu += 1200
    elif continent == 'Oceanie':
        revenu += 1100
    # L'Afrique est la référence (pas de bonus)
    
    # Ajouter un terme d'erreur aléatoire (bruit)
    error_term = random.uniform(-500, 500)
    revenu += error_term
    
    # S'assurer que le revenu est dans des limites raisonnables
    revenu = max(100, min(10000, round(revenu, 1)))
    
    # Ajouter une petite probabilité de valeurs manquantes
    if random.random() < 0.05:
        revenu = np.nan
    
    revenus.append(revenu)

# Ajouter le revenu mensuel calculé au DataFrame
df_micro['RevenuMensuel'] = revenus

# Calculer les dépenses mensuelles basées sur le revenu
depenses = []
for i in range(n):
    revenu = df_micro.loc[i, 'RevenuMensuel']
    if pd.isna(revenu):
        depenses.append(np.nan)
    else:
        # Les dépenses sont environ 70% du revenu, avec une variation
        depense = revenu * random.uniform(0.6, 0.8)
        # Ajouter un terme d'erreur
        depense += random.uniform(-200, 200)
        # S'assurer que les dépenses sont dans des limites raisonnables
        depense = max(50, min(8000, round(depense, 1)))
        depenses.append(depense)

# Ajouter les dépenses mensuelles au DataFrame
df_micro['DepensesMensuelles'] = depenses

# Sauvegarder le DataFrame au format CSV
df_micro.to_csv("donnees2.csv", index=False)

print("✅ Fichier 'donnees2.csv' généré avec 10 000 individus, variables micro, et valeurs manquantes.")
print("   Le RevenuMensuel est maintenant déterminé par les autres variables pour une régression OLS efficace.")