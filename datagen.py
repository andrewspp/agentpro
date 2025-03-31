import pandas as pd
import random

# Liste de 100 vrais noms de pays
noms_pays = [
    "Afghanistan", "Afrique du Sud", "Albanie", "Algérie", "Allemagne", "Andorre", "Angola", "Argentine", "Arménie", "Australie",
    "Autriche", "Azerbaïdjan", "Bahamas", "Bahreïn", "Bangladesh", "Barbade", "Belgique", "Bénin", "Bhoutan", "Biélorussie",
    "Bolivie", "Bosnie-Herzégovine", "Botswana", "Brésil", "Brunei", "Bulgarie", "Burkina Faso", "Burundi", "Cambodge", "Cameroun",
    "Canada", "Cap-Vert", "Chili", "Chine", "Chypre", "Colombie", "Comores", "Congo", "Corée du Sud", "Costa Rica",
    "Côte d'Ivoire", "Croatie", "Cuba", "Danemark", "Djibouti", "Dominique", "Égypte", "Émirats arabes unis", "Équateur", "Érythrée",
    "Espagne", "Estonie", "Eswatini", "États-Unis", "Éthiopie", "Fidji", "Finlande", "France", "Gabon", "Gambie",
    "Géorgie", "Ghana", "Grèce", "Guatemala", "Guinée", "Guyana", "Haïti", "Honduras", "Hongrie", "Inde",
    "Indonésie", "Irak", "Iran", "Irlande", "Islande", "Israël", "Italie", "Jamaïque", "Japon", "Jordanie",
    "Kazakhstan", "Kenya", "Kirghizistan", "Kiribati", "Kosovo", "Koweït", "Laos", "Lettonie", "Liban", "Libéria",
    "Libye", "Liechtenstein", "Lituanie", "Luxembourg", "Macédoine", "Madagascar", "Malaisie", "Malawi", "Maldives", "Mali"
]

def valeur_random(base, variance, mini=None, maxi=None, dec=1):
    val = round(random.uniform(base - variance, base + variance), dec)
    if mini is not None:
        val = max(val, mini)
    if maxi is not None:
        val = min(val, maxi)
    return val

# Données macroéconomiques générées aléatoirement avec des noms de colonnes simplifiés
donnees = {
    "Pays": noms_pays,
    "PIB": [valeur_random(3000, 2000, 200, 25000) for _ in noms_pays],
    "Chomage": [valeur_random(6.0, 4.0, 2.0, 25.0) for _ in noms_pays],
    "Inflation": [valeur_random(3.0, 4.0, -1.0, 20.0) for _ in noms_pays],
    "Population": [valeur_random(50, 1000, 1, 1500) for _ in noms_pays],
    "Balance": [valeur_random(0, 200, -500, 500) for _ in noms_pays],
    "DettePub": [valeur_random(80, 60, 10, 300) for _ in noms_pays],
    "CroissancePIB": [valeur_random(2.0, 3.0, -3.0, 10.0) for _ in noms_pays],
    "IDH": [valeur_random(0.75, 0.15, 0.4, 0.97, 3) for _ in noms_pays],
}

df = pd.DataFrame(donnees)
df.to_csv("donnees.csv", index=False)

print("✅ Fichier 'donnees_macro.csv' généré avec 100 vrais pays et des noms de colonnes simplifiés.")
