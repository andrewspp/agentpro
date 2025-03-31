#!/usr/bin/env python3
import argparse
import json
import logging
import pandas as pd
import subprocess
import sys
import re

from llm_utils import call_llm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent1.log"),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("agent1")

def generate_metadata(csv_file):
    """
    Lit le fichier CSV et extrait les métadonnées détaillées
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"Erreur lors de la lecture du fichier CSV: {e}")
        sys.exit(1)
    
    # Métadonnées basiques
    metadata = {
        "chemin_fichier": csv_file,
        "nb_lignes": len(df),
        "nb_colonnes": len(df.columns),
        "noms_colonnes": list(df.columns),
        "types_colonnes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }
    
    # Statistiques par colonne
    stats = {}
    for col in df.columns:
        col_stats = {
            "valeurs_manquantes": int(df[col].isna().sum()),
            "pourcentage_manquant": float(round((df[col].isna().sum() / len(df)) * 100, 2))
        }
        
        # Statistiques numériques (si applicable)
        if pd.api.types.is_numeric_dtype(df[col]):
            col_stats.update({
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "moyenne": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "mediane": float(df[col].median()) if not pd.isna(df[col].median()) else None,
                "ecart_type": float(df[col].std()) if not pd.isna(df[col].std()) else None,
                "nb_valeurs_uniques": int(df[col].nunique())
            })
        else:  # Statistiques catégorielles
            col_stats.update({
                "nb_valeurs_uniques": int(df[col].nunique()),
                "valeurs_frequentes": df[col].value_counts().head(5).to_dict() if df[col].nunique() < 50 else None
            })
        
        stats[col] = col_stats
    
    metadata["statistiques"] = stats
    return metadata

def detect_data_issues(metadata):
    """
    Identifie les problèmes potentiels dans les données
    """
    issues = []
    
    # Vérification des valeurs manquantes
    for col, stats in metadata["statistiques"].items():
        if stats["pourcentage_manquant"] > 0:
            issues.append({
                "type": "valeurs_manquantes",
                "colonne": col,
                "pourcentage": stats["pourcentage_manquant"],
                "description": f"La colonne '{col}' contient {stats['pourcentage_manquant']}% de valeurs manquantes."
            })
    
    # Détection des valeurs aberrantes pour les colonnes numériques
    for col, stats in metadata["statistiques"].items():
        if "ecart_type" in stats and stats["ecart_type"] is not None:
            if stats["ecart_type"] == 0:
                issues.append({
                    "type": "colonne_constante",
                    "colonne": col,
                    "description": f"La colonne '{col}' contient une valeur constante ({stats['moyenne']})."
                })
    
    # Détection des colonnes à faible variabilité
    for col, stats in metadata["statistiques"].items():
        if stats["nb_valeurs_uniques"] == 1:
            issues.append({
                "type": "variabilite_nulle",
                "colonne": col,
                "description": f"La colonne '{col}' ne contient qu'une seule valeur unique."
            })
    
    return issues

def call_llm_for_problematization(metadata, issues, user_prompt, model, backend):

    """
    Appelle le LLM via le backend choisi (ollama ou gemini)
    pour problématiser les données.
    """
    prompt = f"""## ANALYSE DE DONNÉES ÉCONOMIQUES - PHASE D'INGESTION ET PROBLÉMATISATION

### Métadonnées du fichier
```json
{json.dumps(metadata, indent=2, ensure_ascii=False)}
```

### Problèmes potentiels identifiés
```json
{json.dumps(issues, indent=2, ensure_ascii=False)}
```

### Demande de l'utilisateur
{user_prompt}

---

En tant qu'expert en analyse économétrique et économique, ton rôle est de :

Analyser la structure et la qualité des données.
Problématiser : proposer 1 à 3 questions de recherche claires et pertinentes.
Identifier les limites de ces données (valeurs manquantes, biais potentiels, etc.).
Suggérer des approches économétriques en précisant les méthodes adéquates (corrélation, régression linéaire, différence de différence, regression par discontinuité). Il faut que ce soit adapté aux données
Proposer des pistes de visualisation utiles (au moins 2 ou 3 idées de graphiques).
Précise la nature des variables, discrêtes, continues, catégorielles, etc.
Format attendu :

POINTS DE VIGILANCE (liste des limites/défis)
PROBLÉMATISATION (questions de recherche)
APPROCHES SUGGÉRÉES (méthodes économétriques + idées de visualisation)
Sois précis et complet. 
"""

    logger.info(f"Appel LLM via backend '{backend}' avec modèle '{model}'")
    try:
        return call_llm(prompt=prompt, model_name=model, backend=backend)
    except Exception as e:
        logger.error(f"Erreur lors de l'appel au LLM ({backend}): {e}")
        sys.exit(1)



def parse_llm_output(output):
    """
    Parse la sortie du LLM pour identifier les sections
    """
    sections = {}
    
    # Recherche des sections par regex
    points_pattern = r"(?:##\s*|)POINTS DE VIGILANCE:?([\s\S]*?)(?=##\s*|PROBLÉMATISATION:?|APPROCHES SUGGÉRÉES:?|$)"
    problematisation_pattern = r"(?:##\s*|)PROBLÉMATISATION:?([\s\S]*?)(?=##\s*|POINTS DE VIGILANCE:?|APPROCHES SUGGÉRÉES:?|$)"
    approches_pattern = r"(?:##\s*|)APPROCHES SUGGÉRÉES:?([\s\S]*?)(?=##\s*|POINTS DE VIGILANCE:?|PROBLÉMATISATION:?|$)"
    
    # Extraction des sections
    points_match = re.search(points_pattern, output, re.IGNORECASE)
    if points_match:
        sections["points_vigilance"] = points_match.group(1).strip()
    
    problematisation_match = re.search(problematisation_pattern, output, re.IGNORECASE)
    if problematisation_match:
        sections["problematisation"] = problematisation_match.group(1).strip()
    
    approches_match = re.search(approches_pattern, output, re.IGNORECASE)
    if approches_match:
        sections["approches_suggerees"] = approches_match.group(1).strip()
    
    # Si les sections n'ont pas été trouvées, utiliser le texte entier
    if not sections:
        sections["output_complet"] = output.strip()
    
    return sections

import os

def main():
    parser = argparse.ArgumentParser(
        description="Agent 1: Ingestion des données et problématisation"
    )
    parser.add_argument("csv_file", help="Chemin vers le fichier CSV")
    parser.add_argument("user_prompt", nargs="?", default="",
                        help="Prompt de l'utilisateur (facultatif)")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Modèle LLM à utiliser")
    parser.add_argument("--backend", default="gemini", help="Backend LLM: 'ollama' (local) ou 'gemini'")
    parser.add_argument("--output", default="outputs/agent1_output.json", help="Nom du fichier de sortie JSON")
    args = parser.parse_args()
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Génération des métadonnées
    logger.info(f"Génération des métadonnées pour {args.csv_file}")
    metadata = generate_metadata(args.csv_file)
    
    # Détection des problèmes potentiels dans les données
    logger.info("Détection des problèmes potentiels dans les données")
    issues = detect_data_issues(metadata)
    
    # Appel au LLM pour problématisation
    llm_output = call_llm_for_problematization(metadata, issues, args.user_prompt, args.model, args.backend)
    
    # Parsing de la sortie du LLM
    logger.info("Parsing de la sortie du LLM")
    sections = parse_llm_output(llm_output)
    
    # Préparation de la sortie
    output = {
        "metadata": metadata,
        "data_issues": issues,
        "llm_output": sections,
        "raw_llm_output": llm_output
    }
    
    # Écriture de l'output dans un fichier JSON dans le dossier outputs/
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Output écrit dans {args.output}")
    return 0

if __name__ == "__main__":
    main()

