#!/usr/bin/env python3
"""
Agent 2: Analyse Économétrique

Ce script génère et exécute du code d'analyse économétrique à partir d'un fichier CSV
et des suggestions fournies par l'agent 1. Il gère la correction automatique des erreurs,
capture les visualisations et interprète les résultats.

Usage:
    python agent2.py chemin_csv prompt_utilisateur chemin_sortie_agent1 [--model modele] [--backend backend] [--auto-confirm]

Arguments:
    chemin_csv: Chemin vers le fichier CSV à analyser
    prompt_utilisateur: Prompt initial de l'utilisateur
    chemin_sortie_agent1: Chemin vers le fichier de sortie de l'agent 1
    --model: Modèle LLM à utiliser (défaut: gemma3:27b)
    --backend: Backend LLM ('ollama' ou 'gemini', défaut: 'ollama')
    --auto-confirm: Ignorer la pause manuelle pour correction
"""

import argparse
import json
import logging
import os
import re
import sys
import subprocess
import tempfile
from datetime import datetime
import base64
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
import threading # <--- AJOUTER CECI
import time      # <--- AJOUTER CECI

# Importation du module llm_utils
from llm_utils import call_llm

# ======================================================
# Configuration du logging
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agent2.log"),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("agent2")

def save_prompt_to_file(prompt_content: str, log_file_path: str, prompt_type: str):
    """
    Ajoute un prompt formaté au fichier journal spécifié.
    Crée le répertoire si nécessaire.
    
    Args:
        prompt_content: Contenu du prompt à sauvegarder
        log_file_path: Chemin du fichier journal
        prompt_type: Type de prompt (pour identification)
    """
    try:
        log_dir = os.path.dirname(log_file_path)
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = "=" * 80

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"{separator}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Prompt Type: {prompt_type}\n")
            f.write(f"{separator}\n\n")
            f.write(prompt_content)
            f.write(f"\n\n{separator}\n\n")

        logger.info(f"Prompt '{prompt_type}' ajouté à {log_file_path}")

    except Exception as e:
        logger.error(f"Impossible d'écrire le prompt '{prompt_type}' dans {log_file_path}: {e}")

def interpret_single_visualization_thread(vis, agent1_data, model, backend, prompts_log_path, results_list):
    """
    Fonction exécutée dans un thread pour interpréter une seule visualisation.
    Ajoute le résultat (ou une erreur) à la liste partagée 'results_list'.
    """
    # Utiliser le filename ou l'id comme identifiant unique
    if 'filename' in vis:
        vis_id = os.path.splitext(vis.get("filename", "unknown.png"))[0]
    else:
        vis_id = vis.get("id", f"unknown_{time.time()}") # Fallback ID unique

    try:
        logger.info(f"Thread démarré pour l'interprétation de: {vis_id}")
        interpretation = interpret_visualization_with_gemini(
            vis,
            agent1_data,
            model,
            backend,
            prompts_log_path
        )
        # Ajouter le résultat avec l'identifiant
        results_list.append({'id': vis_id, 'interpretation': interpretation})
        logger.info(f"Interprétation reçue pour {vis_id} dans le thread.")
    except Exception as e:
        logger.error(f"Erreur dans le thread pour {vis_id}: {e}")
        # Ajouter l'erreur avec l'identifiant
        results_list.append({'id': vis_id, 'interpretation': f"Erreur d'interprétation (thread): {e}"})
# --- FIN NOUVELLE FONCTION ---

def extract_code(llm_output: str) -> str:
    """
    Extrait uniquement le code Python des blocs délimités par triple backticks.
    
    Args:
        llm_output: Texte complet de la sortie du LLM
        
    Returns:
        str: Code Python extrait ou texte complet si aucun bloc n'est trouvé
    """
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", llm_output, re.DOTALL)
    if code_blocks:
        return "\n\n".join(code_blocks)
    else:
        return llm_output

def remove_shell_commands(code: str) -> str:
    """
    Filtre le code en supprimant les lignes ressemblant à des commandes shell.
    
    Args:
        code: Code Python potentiellement contenant des commandes shell
        
    Returns:
        str: Code nettoyé
    """
    filtered_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if (stripped.startswith("pip install") or 
            stripped.startswith("bash") or 
            stripped.startswith("$") or 
            (stripped.startswith("python") and not stripped.startswith("python3"))):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines)

def sanitize_code(code: str, csv_path: str, valid_columns: list[str]) -> str:
    """
    Force le bon chemin CSV, corrige les noms de colonnes, corrige les problèmes d'indentation,
    et insère un bloc pour éviter les erreurs de corrélation.
    
    Args:
        code: Code Python à nettoyer
        csv_path: Chemin absolu du fichier CSV
        valid_columns: Liste des noms de colonnes valides
        
    Returns:
        str: Code Python nettoyé
    """
    import ast

    # Forcer le bon chemin CSV
    code = re.sub(
        r"pd\.read_csv\((.*?)\)",
        f"pd.read_csv('{csv_path}')",
        code
    )

    # Identifier les colonnes utilisées
    used_columns = set(re.findall(r"df\[['\"](.*?)['\"]\]", code)) | \
                   set(re.findall(r"df\.([a-zA-Z_][a-zA-Z0-9_]*)", code))

    # Mapper les noms de colonnes utilisés aux noms réels
    col_map = {}
    for used in used_columns:
        for real in valid_columns:
            if used.lower() == real.lower():
                col_map[used] = real
                break

    # Corriger les noms de colonnes
    for used, correct in col_map.items():
        if used != correct:
            code = re.sub(rf"df\[['\"]{used}['\"]\]", f"df['{correct}']", code)
            code = re.sub(rf"df\.{used}\b", f"df['{correct}']", code)

    # Ajouter un bloc pour éviter les erreurs de corrélation
    if "df.corr()" in code or re.search(r"df\.corr\s*\(", code):
        init_numeric_block = "\n# 🔍 Sélection des colonnes numériques pour éviter les erreurs sur df.corr()\ndf_numeric = df.select_dtypes(include='number')\n"
        match = re.search(r"(df\s*=\s*pd\.read_csv\(.*?\))", code)
        if match:
            insertion_point = match.end()
            code = code[:insertion_point] + init_numeric_block + code[insertion_point:]
        else:
            code = init_numeric_block + code

        code = re.sub(r"df\.corr(\s*\(.*?\))", r"df_numeric.corr\1", code)
    
    # NOUVEAU: Corriger les problèmes d'indentation avec plt.show() dans les blocs try/except/else
    # Cette solution utilise une approche ligne par ligne pour une correction précise
    lines = code.split('\n')
    i = 0
    while i < len(lines) - 1:
        current_line = lines[i].rstrip()
        next_line = lines[i + 1].rstrip()
        
        # Chercher plt.show() suivi par except, else ou elif à une indentation différente
        if current_line.strip() == 'plt.show()' and next_line.lstrip().startswith(('except', 'else', 'elif')):
            # Calculer l'indentation du bloc suivant
            next_indent = len(next_line) - len(next_line.lstrip())
            if next_indent > 0:
                # Réindenter plt.show() pour correspondre au bloc
                lines[i] = ' ' * next_indent + 'plt.show()'
        i += 1
    
    # Reconstruire le code avec les lignes corrigées
    code = '\n'.join(lines)
    
    # MODIFICATION: Ne pas désactiver plt.show() complètement, mais le remplacer par une sauvegarde suivie d'un show()
    code = re.sub(r"plt\.show\(\)", "plt.savefig('temp_figure.png', dpi=100, bbox_inches='tight')\nplt.show()", code)
    
    # Remplacer les styles Seaborn obsolètes
    code = code.replace("seaborn-v0_8-whitegrid", "whitegrid")
    code = code.replace("seaborn-whitegrid", "whitegrid")
    code = code.replace("seaborn-v0_8-white", "white")
    code = code.replace("seaborn-white", "white")
    code = code.replace("seaborn-v0_8-darkgrid", "darkgrid")
    code = code.replace("seaborn-darkgrid", "darkgrid")
    code = code.replace("seaborn-v0_8-dark", "dark")
    code = code.replace("seaborn-dark", "dark")
    code = code.replace("seaborn-v0_8-paper", "ticks")
    code = code.replace("seaborn-paper", "ticks")
    code = code.replace("seaborn-v0_8-talk", "ticks")
    code = code.replace("seaborn-talk", "ticks")
    code = code.replace("seaborn-v0_8-poster", "ticks")
    code = code.replace("seaborn-poster", "ticks")
    
    return code

def capture_regression_outputs(output_text, output_dir, executed_code=""):
    """
    Capture les tables de régression OLS du texte de sortie et les enregistre
    à la fois comme texte et comme images. Tente également d'identifier le code
    qui a généré chaque régression.
    
    Args:
        output_text: Texte de sortie contenant potentiellement des résultats de régression
        output_dir: Répertoire où sauvegarder les résultats
        executed_code: Code Python complet qui a été exécuté
        
    Returns:
        Liste des chemins des fichiers générés et des métadonnées associées
    """
    regression_outputs = []
    regression_pattern = r"={10,}\s*\n\s*OLS Regression Results\s*\n={10,}(.*?)(?:\n={10,}|\Z)"
    
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Chercher toutes les tables de régression dans la sortie
    regression_tables = re.findall(regression_pattern, output_text, re.DOTALL)
    
    for i, table_content in enumerate(regression_tables):
        table_id = f"regression_{i+1}"
        
        # Sauvegarder le contenu textuel
        text_file_path = os.path.join(output_dir, f"{table_id}.txt")
        with open(text_file_path, "w", encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("OLS Regression Results\n")
            f.write("="*80 + "\n")
            f.write(table_content)
            f.write("\n" + "="*80 + "\n")
        
        # Créer une visualisation de la table et la sauvegarder comme image
        img_file_path = os.path.join(output_dir, f"{table_id}.png")
        
        # Créer une visualisation de la table avec matplotlib
        plt.figure(figsize=(12, 10))
        plt.text(0.01, 0.99, "OLS Regression Results\n" + table_content, 
                 family='monospace', fontsize=10, 
                 verticalalignment='top')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(img_file_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Encoder l'image en base64 pour l'inclure dans la sortie JSON
        with open(img_file_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Extraire quelques métadonnées de base (R-squared, variables clés)
        r_squared_match = re.search(r"R-squared:\s*([\d\.]+)", table_content)
        r_squared = r_squared_match.group(1) if r_squared_match else "N/A"
        
        key_variables = re.findall(r"([A-Za-z0-9_]+)\s+[\d\.-]+\s+[\d\.-]+\s+[\d\.-]+\s+[\d\.-]+", table_content)
        
        # Extraire quelques statistiques clés pour les données de régression
        regression_data = {
            "r_squared": r_squared,
            "variables": key_variables[:5],  # Limiter aux 5 premières variables
            "statistics": {}
        }

        coef_section = re.search(r"==+\s*\n\s*(coef.*?)(?:\n\s*==+|\Z)", output_text, re.DOTALL)
        if coef_section:
            lines = coef_section.group(1).split('\n')
            headers = None
            coef_data = []
            
            for line in lines:
                if "coef" in line.lower():
                    # C'est la ligne d'en-tête
                    headers = re.split(r'\s{2,}', line.strip())
                elif line.strip() and headers:
                    # C'est une ligne de données
                    values = re.split(r'\s{2,}', line.strip())
                    if len(values) >= len(headers):
                        variable = values[0]
                        coef_data.append({
                            "variable": variable,
                            "coef": values[1] if len(values) > 1 else "N/A",
                            "std_err": values[2] if len(values) > 2 else "N/A",
                            "p_value": values[4] if len(values) > 4 else "N/A"
                        })
            
            regression_data["coefficients"] = coef_data
            
            # NOUVELLE SECTION: Convertir les données de régression en CSV amélioré pour faciliter l'interprétation
            try:
                import pandas as pd
                if coef_data:
                    coef_df = pd.DataFrame(coef_data)
                    
                    # Ajouter une colonne de significativité pour faciliter l'interprétation
                    try:
                        coef_df['significatif'] = coef_df['p_value'].astype(float) < 0.05
                    except:
                        # Si conversion en float échoue, on continue sans cette colonne
                        pass
                    
                    # Ajouter des métadonnées contextuelles en commentaire
                    csv_header = f"# Résultats de régression - R²: {r_squared}\n"
                    csv_header += f"# Interprétation: un coefficient positif indique une relation positive, une p-value < 0.05 indique une significativité statistique\n"
                    csv_header += f"# Variables explicatives: {', '.join(key_variables[:5]) if key_variables else 'non disponibles'}\n"
                    
                    regression_data["csv_data"] = csv_header + coef_df.to_csv(index=False)
                    
                    # Sauvegarder en fichier CSV
                    csv_path = os.path.join(output_dir, f"{table_id}_coefficients.csv")
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        f.write(regression_data["csv_data"])
                    logger.info(f"Données CSV améliorées de coefficients sauvegardées: {csv_path}")
            except Exception as e:
                logger.error(f"Erreur lors de la conversion des coefficients en CSV amélioré: {e}")
        
        # Extraire le code de régression si le code complet est fourni
        regression_code = ""
        if executed_code:
            regression_code = extract_regression_code(executed_code, table_content)
            regression_data["regression_code"] = regression_code
        
        regression_outputs.append({
            'type': 'regression_table',
            'id': table_id,
            'text_path': text_file_path,
            'image_path': img_file_path,
            'base64': img_data,
            'metadata': {
                'r_squared': r_squared,
                'variables': key_variables[:5]  # Limiter aux 5 premières variables
            },
            'data': regression_data,  # Ajout des données structurées
            'csv_data': regression_data.get("csv_data", ""),  # Ajout des données CSV améliorées
            'regression_code': regression_code  # Ajout du code de régression
        })
    
    return regression_outputs

def interpret_visualization_with_gemini(vis, agent1_data, model, backend, prompts_log_path, timeout):
    """
    Interprète une visualisation en envoyant directement l'image à Gemini via la fonction call_llm.
    
    Args:
        vis: Métadonnées de la visualisation avec base64 de l'image
        agent1_data: Données de l'agent1
        model: Modèle LLM à utiliser
        backend: Backend pour les appels LLM
        prompts_log_path: Chemin pour sauvegarder les prompts
        
    Returns:
        str: Interprétation de la visualisation
    """
    # Importer call_llm depuis llm_utils
    from llm_utils import call_llm
    
    # Valider que l'image est disponible
    if 'base64' not in vis or not vis['base64']:
        logger.error(f"Pas de données base64 disponibles pour la visualisation {vis.get('id', 'non identifiée')}")
        return "Erreur: Impossible d'analyser cette visualisation (image non disponible)"
    
    # Extraire le titre et le type
    if 'filename' in vis:
        filename = vis.get("filename", "figure.png")
        vis_id = os.path.splitext(filename)[0]
    else:
        vis_id = vis.get("id", "visualisation")
        filename = vis_id + '.png'
    
    # Déterminer le type de visualisation
    vis_type = "Unknown visualization"
    if 'regression' in vis_id.lower():
        vis_type = "Table de Régression OLS"
    elif 'correlation' in vis_id.lower() or 'corr' in vis_id.lower():
        vis_type = "Matrice de Corrélation"
    elif 'distribution' in vis_id.lower() or 'hist' in vis_id.lower():
        vis_type = "Graphique de Distribution"
    elif 'scatter' in vis_id.lower() or 'relation' in vis_id.lower():
        vis_type = "Nuage de Points"
    elif 'box' in vis_id.lower():
        vis_type = "Boîte à Moustaches"
    
    # Récupérer le titre
    title = vis.get("title", vis_type)
    
    # Extraire des informations supplémentaires pour le contexte
    metadata_str = json.dumps(agent1_data["metadata"], indent=2, ensure_ascii=False)
    extra_context = ""
    if 'metadata' in vis:
        if 'r_squared' in vis['metadata']:
            extra_context += f"\nR-squared: {vis['metadata']['r_squared']}"
        if 'variables' in vis['metadata']:
            extra_context += f"\nVariables principales: {', '.join(vis['metadata']['variables'])}"
    
    # Créer le prompt pour Gemini
    prompt = f"""## INTERPRÉTATION DE VISUALISATION ÉCONOMIQUE

### Type de visualisation
{vis_type}

### Titre
{title}

### Identifiant
{vis_id}

### Métadonnées spécifiques
{extra_context}

### Contexte des données
Le dataset contient les variables suivantes: {', '.join(agent1_data["metadata"].get("noms_colonnes", []))}

### Métadonnées de l'ensemble de données
```json
{metadata_str[:500]}...
```

### Question de recherche initiale
{agent1_data.get("user_prompt", "Non disponible")}

---

Analyse cette visualisation économique. Tu reçois directement l'image, donc base ton analyse sur ce que tu observes visuellement. Ton interprétation doit:

1. Décrire précisément ce que montre la visualisation (tendances, relations, valeurs aberrantes)
2. Expliquer les relations entre les variables visibles
3. Mentionner les valeurs numériques spécifiques (minimums, maximums, moyennes) que tu peux déduire visuellement
4. Relier cette visualisation à la question de recherche

Ton interprétation doit être factuelle, précise et basée uniquement sur ce que tu peux observer dans l'image. Reste économique dans ton analyse en te concentrant sur les informations les plus importantes.
"""

    # Sauvegarder le prompt dans le fichier journal
    save_prompt_to_file(prompt, prompts_log_path, f"Gemini Image Interpretation - {vis_id}")
    
    try:
        # Utiliser call_llm de llm_utils qui gère maintenant les images
        logger.info(f"Appel à Gemini avec image pour visualisation {vis_id}")
        interpretation = call_llm(
            prompt=prompt, 
            model_name=model, 
            backend=backend, 
            image_base64=vis['base64']
        )
        
        logger.info(f"Interprétation générée par Gemini pour: {vis_id}")
        return interpretation
    except Exception as e:
        logger.error(f"Erreur lors de l'interprétation de l'image pour {vis_id}: {e}")
        return f"Erreur d'interprétation: {e}"

# --- REMPLACEMENT DE LA FONCTION ---
def generate_visualization_interpretations(visualizations, regression_outputs, agent1_data, model, backend, prompts_log_path, timeout=120):
    """
    Génère des interprétations pour les visualisations SÉQUENTIELLEMENT (SANS THREADING).
    Utilise le LLM Gemini directement à partir des images.

    Args:
        visualizations (list): Liste des métadonnées des visualisations.
        regression_outputs (list): Liste des sorties des tables de régression.
        agent1_data (dict): Données de l'agent1.
        model (str): Modèle LLM à utiliser.
        backend (str): Backend pour les appels LLM.
        prompts_log_path (str): Chemin pour sauvegarder les prompts.
        timeout (int): Timeout en secondes pour chaque appel LLM d'interprétation. Défaut: 120.

    Returns:
        Liste mise à jour des visualisations avec interprétations.
    """
    # Combiner visualisations et régressions
    all_visuals = visualizations + regression_outputs
    if not all_visuals:
        logger.info("Aucune visualisation ou table de régression à interpréter.")
        return []

    logger.info(f"Lancement des interprétations SÉQUENTIELLES pour {len(all_visuals)} éléments (timeout/item: {timeout}s)")

    updated_visuals = [] # Nouvelle liste pour stocker les résultats mis à jour

    for i, vis in enumerate(all_visuals):
        # Générer un ID unique si nécessaire pour le mapping des résultats
        if 'id' not in vis: # Assigner un ID si manquant
             vis_prefix = os.path.splitext(vis.get("filename", "unknown"))[0] if 'filename' in vis else vis.get("type", "item")
             vis['id'] = f"{vis_prefix}_{i}_{int(time.time()*1000)}"

        vis_id = vis['id'] # Utiliser l'ID assigné

        # Vérifier que l'image est disponible en base64
        if 'base64' not in vis or not vis['base64']:
            logger.warning(f"({i+1}/{len(all_visuals)}) Image manquante pour {vis_id}. Interprétation impossible.")
            vis['interpretation'] = "Interprétation impossible: image non disponible"
            updated_visuals.append(vis) # Ajouter l'élément avec l'erreur
            continue # Passe à la visualisation suivante

        logger.info(f"({i+1}/{len(all_visuals)}) Interprétation de : {vis_id}")
        try:
            # Appel DIRECT à la fonction d'interprétation
            interpretation = interpret_visualization_with_gemini(
                vis,
                agent1_data,
                model,
                backend,
                prompts_log_path,
                timeout # Passer le timeout
            )
            vis['interpretation'] = interpretation
            logger.info(f"({i+1}/{len(all_visuals)}) Interprétation reçue pour {vis_id}.")

        except Exception as e:
            logger.error(f"({i+1}/{len(all_visuals)}) Erreur lors de l'interprétation directe pour {vis_id}: {e}", exc_info=True)
            vis['interpretation'] = f"Erreur d'interprétation (directe): {e}"

        updated_visuals.append(vis) # Ajouter l'élément traité (avec succès ou erreur)

    logger.info(f"Toutes les {len(all_visuals)} interprétations séquentielles sont terminées.")

    return updated_visuals

# --- FIN REMPLACEMENT FONCTION ---
def attempt_execution_loop(code: str, csv_file: str, agent1_data: dict, model: str, prompts_log_path: str, backend: str) -> dict:
    """
    Tente d'exécuter le code et, en cas d'erreur, demande une correction au LLM en utilisant le backend choisi.
    
    Args:
        code: Code Python à exécuter
        csv_file: Chemin absolu du fichier CSV
        agent1_data: Données de l'agent1
        model: Modèle LLM à utiliser
        prompts_log_path: Chemin pour sauvegarder les prompts
        backend: Backend pour les appels LLM ('ollama' ou 'gemini')
        
    Returns:
        Dictionnaire contenant les résultats de l'exécution
    """
    import base64
    from collections import deque
    import time
    import shutil
    import os
    import tempfile
    import json
    import subprocess
    import logging
    from datetime import datetime

    logger = logging.getLogger("agent2")

    # Initialiser le modèle courant (sera mis à jour si on bascule vers le modèle puissant)
    current_model = model

    # Vérifier si le code utilise le bon chemin CSV
    if csv_file not in code:
        logger.warning(f"Le code initial ne semble pas utiliser le chemin absolu '{csv_file}'.")

    # Créer les répertoires temporaires pour l'exécution
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = tempfile.mkdtemp(prefix=f"analysis_{timestamp}_")
    
    # Ajouter un répertoire pour les tables de régression
    tables_dir = os.path.join(temp_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    # Ajouter un répertoire pour les visualisations
    vis_dir = os.path.join(temp_dir, "visualisations")
    os.makedirs(vis_dir, exist_ok=True)

    # Créer un répertoire pour sauvegarder les versions du code
    code_versions_dir = os.path.join("outputs", f"code_versions_{timestamp}")
    os.makedirs(code_versions_dir, exist_ok=True)
    logger.info(f"Les versions de code seront sauvegardées dans {code_versions_dir}")

    import textwrap

# Code pour capturer les visualisations - MODIFIÉ AVEC SUPPORT CSV AMÉLIORÉ
    vis_code = textwrap.dedent(f"""
# Ajout pour capturer les visualisations
import os
import logging
import io
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import json
import numpy as np

# Configuration pour un meilleur rendu - Utilisation d'un style compatible avec les versions récentes de matplotlib
try:
    plt.style.use('whitegrid')  # Pour seaborn récent
except:
    try:
        plt.style.use('seaborn')  # Fallback vers le style seaborn de base
    except:
        pass  # Utiliser le style par défaut si seaborn n'est pas disponible

# Définition des répertoires de sortie
VIS_DIR = r"{vis_dir}"
TABLES_DIR = r"{tables_dir}"
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# Configuration du logger
vis_logger = logging.getLogger("visualisation_capture")

# Redirection de l'affichage pour capturer les sorties
from io import StringIO
import sys

# Classe pour capturer stdout
class CaptureOutput:
    def __init__(self):
        self.old_stdout = sys.stdout
        self.buffer = StringIO()

    def __enter__(self):
        sys.stdout = self.buffer
        return self.buffer

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.old_stdout

# Classe pour convertir les données numpy en JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super(NumpyEncoder, self).default(obj)

# Remplacer la fonction plt.show() pour enregistrer les figures
_original_show = plt.show
_fig_counter = 0

def _custom_show(*args, **kwargs):
    global _fig_counter
    try:
        fig = plt.gcf()
        if fig.get_axes():
            _fig_counter += 1
            filepath = os.path.join(VIS_DIR, f"figure_{{_fig_counter}}.png")

            # Capturer les données du graphique actuel
            fig_data = {{}}

            # MODIFICATION: Capturer les informations sur les axes et le titre
            fig_title = fig.get_label() or f"Figure {{_fig_counter}}"
            fig_data["title"] = fig_title

            # MODIFICATION: Essayer de capturer les données utilisées pour la visualisation
            for ax_idx, ax in enumerate(fig.get_axes()):
                ax_data = {{}}
                
                # Capturer les titres et labels d'axes
                x_label = ax.get_xlabel() or "Axe X"
                y_label = ax.get_ylabel() or "Axe Y"
                ax_title = ax.get_title() or fig_title
                
                ax_data["x_label"] = x_label
                ax_data["y_label"] = y_label
                ax_data["title"] = ax_title
                
                # Capturer les données des lignes
                for line_idx, line in enumerate(ax.get_lines()):
                    line_id = f"ax{{ax_idx}}_line{{line_idx}}"
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    
                    # Créer un petit dataframe et le convertir en CSV avec contexte
                    if len(x_data) > 0 and len(y_data) > 0:
                        try:
                            import pandas as pd
                            # Utiliser les labels des axes comme noms de colonnes
                            line_df = pd.DataFrame({{
                                x_label: x_data, 
                                y_label: y_data
                            }})
                            
                            # CORRECTION: Récupérer le titre de l'axe depuis ax_data ou utiliser une valeur par défaut
                            current_ax_title = ax_data.get("title", f"Figure {{_fig_counter}}")
                            
                            # Ajouter des métadonnées en commentaire CSV
                            csv_header = f"# {{current_ax_title}} - Relation entre {{x_label}} et {{y_label}}\\n"  # Ajout du caractère \\n à la fin
                            csv_header += f"# Source: Figure {{_fig_counter}}, Ligne {{line_idx+1}}\\n"
                            
                            # Ajouter des statistiques en commentaire CSV
                            if hasattr(x_data, 'min') and hasattr(y_data, 'min'):
                                stats = f"# Statistiques {{x_label}}: min={{x_data.min():.2f}}, max={{x_data.max():.2f}}, moyenne={{x_data.mean():.2f}}\\n"
                                stats += f"# Statistiques {{y_label}}: min={{y_data.min():.2f}}, max={{y_data.max():.2f}}, moyenne={{y_data.mean():.2f}}\\n"
                                csv_header += stats
                            
                            csv_data = csv_header + line_df.to_csv(index=False)
                            
                            ax_data[line_id] = {{
                                "type": "line",
                                "x": x_data.tolist() if hasattr(x_data, 'tolist') else list(x_data),
                                "y": y_data.tolist() if hasattr(y_data, 'tolist') else list(y_data),
                                "label": line.get_label() if line.get_label() != '_nolegend_' else f"Line {{line_idx+1}}",
                                "x_label": x_label,  # Ajout du label de l'axe x
                                "y_label": y_label,  # Ajout du label de l'axe y
                                "title": current_ax_title,   # Ajout du titre (corrigé)
                                "csv_data": csv_data
                            }}
                        except Exception as e:
                            vis_logger.error(f"Erreur lors de la conversion des données en CSV amélioré: {{e}}")
                            ax_data[line_id] = {{
                                "type": "line",
                                "x": x_data.tolist() if hasattr(x_data, 'tolist') else list(x_data),
                                "y": y_data.tolist() if hasattr(y_data, 'tolist') else list(y_data),
                                "label": line.get_label() if line.get_label() != '_nolegend_' else f"Line {{line_idx+1}}",
                                "x_label": x_label,
                                "y_label": y_label
                            }}

                # Extraire les lignes, barres, etc.
                for line_idx, line in enumerate(ax.get_lines()):
                    line_id = f"ax{{ax_idx}}_line{{line_idx}}"
                    if line_id not in ax_data:  # Si n'est pas déjà ajouté avec CSV
                        ax_data[line_id] = {{
                            "type": "line",
                            "x": line.get_xdata().tolist() if hasattr(line, 'get_xdata') else [],
                            "y": line.get_ydata().tolist() if hasattr(line, 'get_ydata') else [],
                            "label": line.get_label() if line.get_label() != '_nolegend_' else f"Line {{line_idx+1}}",
                            "x_label": x_label,
                            "y_label": y_label
                        }}

                # Barres
                for container_idx, container in enumerate(ax.containers):
                    if isinstance(container, matplotlib.container.BarContainer):
                        container_id = f"ax{{ax_idx}}_bars{{container_idx}}"
                        rectangles = container.get_children()
                        bar_data = {{
                            "type": "bar",
                            "heights": [],
                            "positions": [],
                            "widths": [],
                            "x_label": x_label,
                            "y_label": y_label
                        }}
                        for rect in rectangles:
                            if hasattr(rect, 'get_height') and hasattr(rect, 'get_x') and hasattr(rect, 'get_width'):
                                bar_data["heights"].append(rect.get_height())
                                bar_data["positions"].append(rect.get_x())
                                bar_data["widths"].append(rect.get_width())
                        ax_data[container_id] = bar_data
                        
                        # Essayer de convertir les données de barres en CSV avec contexte
                        try:
                            import pandas as pd
                            
                            # Créer un DataFrame pour les barres avec noms personnalisés
                            bar_df = pd.DataFrame({{
                                x_label: bar_data["positions"],
                                f"{{y_label}} (hauteur)": bar_data["heights"],
                                "Largeur": bar_data["widths"]
                            }})
                            
                            # CORRECTION: Récupérer le titre de l'axe depuis ax_data ou utiliser une valeur par défaut
                            current_ax_title = ax_data.get("title", f"Figure {{_fig_counter}}")
                            
                            # Ajouter des métadonnées en commentaire CSV
                            csv_header = f"# {{current_ax_title}} - Graphique à barres\\n"
                            csv_header += f"# Source: Figure {{_fig_counter}}, Container {{container_idx+1}}\\n"
                            csv_header += f"# X: {{x_label}}, Y: {{y_label}}\\n"
                            
                            # Ajouter des statistiques
                            if len(bar_data["heights"]) > 0:
                                heights = np.array(bar_data["heights"])
                                stats = f"# Statistiques hauteurs: min={{heights.min():.2f}}, max={{heights.max():.2f}}, moyenne={{heights.mean():.2f}}\\n"
                                csv_header += stats
                            
                            ax_data[container_id]["csv_data"] = csv_header + bar_df.to_csv(index=False)
                        except Exception as e:
                            vis_logger.error(f"Erreur lors de la conversion des données de barres en CSV amélioré: {{e}}")

                # Collections (scatter plots, heatmaps)
                for collection_idx, collection in enumerate(ax.collections):
                    collection_id = f"ax{{ax_idx}}_collection{{collection_idx}}"
                    collection_data = {{"type": "collection", "x_label": x_label, "y_label": y_label}}

                    # Points (scatter)
                    if hasattr(collection, 'get_offsets'):
                        offsets = collection.get_offsets()
                        if len(offsets) > 0:
                            collection_data["points"] = offsets.tolist() if hasattr(offsets, 'tolist') else []
                            
                            # Convertir en CSV amélioré avec contexte
                            try:
                                import pandas as pd
                                offsets_array = np.array(offsets)
                                if offsets_array.shape[1] == 2:  # Si ce sont des points 2D
                                    # Utiliser les labels des axes comme noms de colonnes
                                    points_df = pd.DataFrame({{
                                        x_label: offsets_array[:, 0],
                                        y_label: offsets_array[:, 1]
                                    }})
                                    
                                    # CORRECTION: Récupérer le titre de l'axe depuis ax_data ou utiliser une valeur par défaut
                                    current_ax_title = ax_data.get("title", f"Figure {{_fig_counter}}")
                                    
                                    # Ajouter des métadonnées en commentaire CSV
                                    csv_header = f"# {{current_ax_title}} - Nuage de points\\n"
                                    csv_header += f"# Source: Figure {{_fig_counter}}, Collection {{collection_idx+1}}\\n"
                                    
                                    # Ajouter des statistiques
                                    x_stats = f"# Statistiques {{x_label}}: min={{offsets_array[:, 0].min():.2f}}, max={{offsets_array[:, 0].max():.2f}}, moyenne={{offsets_array[:, 0].mean():.2f}}\\n"
                                    y_stats = f"# Statistiques {{y_label}}: min={{offsets_array[:, 1].min():.2f}}, max={{offsets_array[:, 1].max():.2f}}, moyenne={{offsets_array[:, 1].mean():.2f}}\\n"
                                    csv_header += x_stats + y_stats
                                    
                                    collection_data["csv_data"] = csv_header + points_df.to_csv(index=False)
                            except Exception as e:
                                vis_logger.error(f"Erreur lors de la conversion des points en CSV amélioré: {{e}}")

                    # Couleurs
                    if hasattr(collection, 'get_array') and collection.get_array() is not None:
                        collection_data["values"] = collection.get_array().tolist() if hasattr(collection.get_array(), 'tolist') else []

                    ax_data[collection_id] = collection_data

                # Tentative d'extraire les étiquettes des axes
                if ax.get_title():
                    ax_data["title"] = ax.get_title()
                if ax.get_xlabel():
                    ax_data["xlabel"] = ax.get_xlabel()
                if ax.get_ylabel():
                    ax_data["ylabel"] = ax.get_ylabel()

                # Tentative d'extraire les limites des axes
                if ax.get_xlim():
                    ax_data["xlim"] = ax.get_xlim()
                if ax.get_ylim():
                    ax_data["ylim"] = ax.get_ylim()

                fig_data[f"axes{{ax_idx}}"] = ax_data

            # Information globale sur la figure
            if fig.get_label():
                fig_data["title"] = fig.get_label()
            else:
                fig_data["title"] = f"Figure {{_fig_counter}}"
                
            # Essayer d'extraire directement un dataframe de la figure si possible
            fig_csv_data = ""
            try:
                # Parcourir les objets pour trouver un DataFrame
                for ax in fig.get_axes():
                    for line in ax.get_lines():
                        if hasattr(line, '_data_source') and hasattr(line._data_source, 'to_csv'):
                            title = fig_data.get("title", "Figure")
                            x_label = ax.get_xlabel() or "X"
                            y_label = ax.get_ylabel() or "Y"
                            
                            # Créer un en-tête CSV avec métadonnées
                            csv_header = f"# {{title}} - Données source\\n"
                            csv_header += f"# Axes: {{x_label}} vs {{y_label}}\\n"
                            # Ajouter des statistiques si possible
                            if hasattr(line._data_source, 'describe'):
                                try:
                                    describe = line._data_source.describe().to_dict()
                                    for col, stats in describe.items():
                                        csv_header += f"# Statistiques {{col}}: min={{stats.get('min', 'N/A')}}, max={{stats.get('max', 'N/A')}}, moyenne={{stats.get('mean', 'N/A')}}\\n"
                                except:
                                    pass
                            
                            fig_csv_data = csv_header + line._data_source.to_csv(index=False)
                            break
            except Exception as e:
                vis_logger.error(f"Erreur lors de l'extraction directe du DataFrame: {{e}}")
            
            if fig_csv_data:
                fig_data["csv_data"] = fig_csv_data

            try:
                plt.savefig(filepath, bbox_inches='tight', dpi=100)
                vis_logger.info(f"Visualisation sauvegardée : {{filepath}}")

                # Sauvegarde des données dans un fichier JSON
                data_filepath = os.path.join(VIS_DIR, f"figure_{{_fig_counter}}_data.json")
                with open(data_filepath, 'w', encoding='utf-8') as f:
                    json.dump(fig_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
                vis_logger.info(f"Données de visualisation sauvegardées : {{data_filepath}}")
                
                # Sauvegarde des données CSV si disponibles
                if any(["csv_data" in data for ax_data in fig_data.values() if isinstance(ax_data, dict) 
                       for data in ax_data.values() if isinstance(data, dict)]) or "csv_data" in fig_data:
                    csv_filepath = os.path.join(VIS_DIR, f"figure_{{_fig_counter}}_data.csv")
                    
                    # Choisir la première source de données CSV disponible
                    csv_content = fig_data.get("csv_data", "")
                    if not csv_content:
                        for ax_data in fig_data.values():
                            if isinstance(ax_data, dict):
                                for data in ax_data.values():
                                    if isinstance(data, dict) and "csv_data" in data:
                                        csv_content = data["csv_data"]
                                        break
                                if csv_content:
                                    break
                    
                    if csv_content:
                        with open(csv_filepath, 'w', encoding='utf-8') as f:
                            f.write(csv_content)
                        vis_logger.info(f"Données CSV améliorées sauvegardées : {{csv_filepath}}")

            except Exception as e:
                vis_logger.error(f"Impossible de sauvegarder la figure {{_fig_counter}} à {{filepath}}: {{e}}")

            # On permet l'affichage pour débogage mais ça sera ignoré dans l'environnement non-interactif
            _original_show(*args, **kwargs)
    except Exception as e:
        vis_logger.error(f"Erreur dans _custom_show: {{e}}")

plt.show = _custom_show

# Fonction pour capturer les sorties de régression OLS de statsmodels
_original_print = print
def _custom_print(*args, **kwargs):
    output = " ".join(str(arg) for arg in args)
    _original_print(*args, **kwargs)  # Affiche normalement

    # Détecter si c'est un résultat de régression OLS
    if "OLS Regression Results" in output and "=" * 10 in output:
        try:
            # Extraire la table de régression complète
            global _tables_counter
            if '_tables_counter' not in globals():
                _tables_counter = 0
            _tables_counter += 1

            # Sauvegarder le texte
            filepath = os.path.join(TABLES_DIR, f"regression_{{_tables_counter}}.txt")
            with open(filepath, "w") as f:
                f.write(output)

            # Créer une visualisation de la table
            img_path = os.path.join(TABLES_DIR, f"regression_{{_tables_counter}}.png")
            plt.figure(figsize=(12, 10))
            plt.text(0.01, 0.99, output, family='monospace', fontsize=10, verticalalignment='top')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(img_path, dpi=100, bbox_inches='tight')
            plt.close()

            # Extraire et sauvegarder les données de régression
            # Extraire R-squared
            r_squared_match = re.search(r"R-squared:\\s*([\d\.]+)", output)
            r_squared = r_squared_match.group(1) if r_squared_match else "N/A"

            # Extraire les variables et coefficients
            coef_section = re.search(r"==+\\s*\\n\\s*(coef.*?)(?:\\n\\s*==+|\\Z)", output, re.DOTALL)
            regression_data = {{
                "r_squared": r_squared,
                "coefficients": []
            }}

            if coef_section:
                lines = coef_section.group(1).split('\\n') # Utilisation de \\n car c'est dans une string Python
                headers = None

                for line in lines:
                    if "coef" in line.lower():
                        # C'est la ligne d'en-tête
                        headers = re.split(r'\\s{{2,}}', line.strip()) # Utilisation de \\s{{2,}}
                    elif line.strip() and headers:
                        # C'est une ligne de données
                        values = re.split(r'\\s{{2,}}', line.strip()) # Utilisation de \\s{{2,}}
                        if len(values) >= len(headers):
                            variable = values[0]
                            coef_data = {{
                                "variable": variable,
                                "coef": values[1] if len(values) > 1 else "N/A",
                                "std_err": values[2] if len(values) > 2 else "N/A",
                                "p_value": values[4] if len(values) > 4 else "N/A"
                            }}
                            regression_data["coefficients"].append(coef_data)
                            
            # NOUVELLE SECTION: Convertir les coefficients en CSV amélioré
            try:
                import pandas as pd
                if regression_data["coefficients"]:
                    coef_df = pd.DataFrame(regression_data["coefficients"])
                    
                    # Ajouter une colonne de significativité pour faciliter l'interprétation
                    try:
                        coef_df['significatif'] = coef_df['p_value'].astype(float) < 0.05
                    except:
                        # Si conversion en float échoue, on continue sans cette colonne
                        pass
                    
                    # Ajouter des métadonnées contextuelles en commentaire
                    csv_header = f"# Résultats de régression OLS - Table {{_tables_counter}}\\n"
                    csv_header += f"# R²: {{r_squared}}\\n"
                    csv_header += f"# Interprétation: un coefficient positif indique une relation positive avec la variable dépendante\\n"
                    csv_header += f"# Une p-value < 0.05 indique que le coefficient est statistiquement significatif\\n"
                    
                    regression_data["csv_data"] = csv_header + coef_df.to_csv(index=False)
                    
                    # Sauvegarder aussi en fichier CSV amélioré
                    csv_path = os.path.join(TABLES_DIR, f"regression_{{_tables_counter}}_coefficients.csv")
                    with open(csv_path, 'w', encoding='utf-8') as f:
                        f.write(regression_data["csv_data"])
                    vis_logger.info(f"Données CSV améliorées de coefficients sauvegardées: {{csv_path}}")
            except Exception as e:
                vis_logger.error(f"Erreur lors de la conversion des coefficients en CSV amélioré: {{e}}")

            # Sauvegarder les données
            data_path = os.path.join(TABLES_DIR, f"regression_{{_tables_counter}}_data.json")
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump(regression_data, f, ensure_ascii=False, indent=2)

            vis_logger.info(f"Table de régression sauvegardée : {{filepath}}")
            vis_logger.info(f"Données de régression sauvegardées : {{data_path}}")
        except Exception as e:
            vis_logger.error(f"Erreur lors de la capture d'une table de régression: {{e}}")

# Remplacer la fonction print
print = _custom_print

# Fonction manuelle pour sauvegarder des figures
def save_figure(fig, name):
    filepath = os.path.join(VIS_DIR, f"{{name}}.png")
    fig.savefig(filepath, bbox_inches='tight', dpi=100)
    vis_logger.info(f"Figure sauvegardée manuellement: {{filepath}}")
    return filepath
""") # Fin de la string vis_code


    # Initialiser les variables pour la boucle d'exécution
    temp_filename = os.path.join(temp_dir, "analysis_script.py")
    attempt = 0
    llm_correction_attempt = 0  # Compteur spécifique pour les tentatives de correction par LLM
    execution_results = {"success": False}
    execution_output = ""
    all_code_versions = []
    recent_errors = deque(maxlen=3)
    max_attempts = 10
    current_stderr = ""

    # Flag pour le modèle puissant
    tried_powerful_model = False
    
    # Boucle principale d'exécution
    while attempt < max_attempts:
        attempt += 1
        logger.info(f"--- Tentative d'exécution {attempt}/{max_attempts} ---")

        # Sanitize le code avant l'exécution
        try:
            valid_columns = agent1_data["metadata"].get("noms_colonnes", [])
            sanitized_code = sanitize_code(code, csv_file, valid_columns)
            if sanitized_code != code:
                logger.info("Code assaini (chemin CSV, colonnes, df_numeric).")
                code = sanitized_code
        except Exception as e:
            logger.error(f"Erreur pendant l'assainissement du code: {e}. Tentative avec le code actuel.")

        # Sauvegarder la version actuelle du code
        current_code_path = os.path.join(code_versions_dir, f"attempt_{attempt}.py")
        try:
            with open(current_code_path, "w", encoding="utf-8") as f:
                f.write(code)
            all_code_versions.append(current_code_path)
            logger.info(f"Version {attempt} du code sauvegardée: {current_code_path}")
        except Exception as e:
             logger.error(f"Impossible de sauvegarder la version {attempt} du code : {e}")

        # Par celles-ci:
        import textwrap
        vis_code_dedented = textwrap.dedent(vis_code)  # Conserve l'indentation relative
        modified_code = vis_code_dedented + "\n\n" + code.strip()

        # Écrire le code modifié dans un fichier temporaire
        try:
            with open(temp_filename, "w", encoding="utf-8") as f:
                f.write(modified_code)
        except Exception as e:
            logger.error(f"Impossible d'écrire le script temporaire {temp_filename}: {e}")
            execution_results = {"success": False, "error": "Erreur écriture fichier temporaire", "stderr": str(e)}
            break

                            # <<< AJOUTER CE BLOC DE DÉBOGAGE >>>
        logger.info(f"--- DEBUG: Vérification du contenu écrit dans {temp_filename} ---")
        try:
            with open(temp_filename, "r", encoding="utf-8") as f_read:
                lines_to_check = 5
                lines = []
                for i in range(lines_to_check):
                    try:
                        lines.append(next(f_read).rstrip('\n'))
                    except StopIteration:
                        break # Fin du fichier
                logger.info(f"Premières {len(lines)} lignes (espaces représentés par '·', tabulations par '\\t'):")
                for i, line in enumerate(lines):
                    # Représentation visuelle des espaces/tabs
                    line_repr = line.replace(' ', '·').replace('\t', '\\t') 
                    logger.info(f"  Ligne {i+1}: [{line_repr}]")
        except Exception as read_err:
            logger.error(f"  Erreur de lecture du fichier temporaire pour débogage: {read_err}")
        logger.info("--- FIN DEBUG ---")
        # <<< FIN DU BLOC DE DÉBOGAGE >>>


        # Exécuter le code
        try:
            logger.info(f"Exécution de: python3 {temp_filename}")
            result = subprocess.run(
                ["python3", temp_filename],
                capture_output=True, text=True, check=True, timeout=300
            )
            logger.info("Exécution réussie")
            logger.debug(f"Sortie standard (stdout):\n{result.stdout}")
            execution_output = result.stdout
            execution_results = {
                "success": True,
                "output": result.stdout,
                "temp_dir": temp_dir,
                "vis_dir": vis_dir,
                "tables_dir": tables_dir,
                "script_path": temp_filename,
                "all_code_versions": all_code_versions,
                "final_code_used_path": current_code_path
            }

            # Traiter les visualisations
            vis_files = []
            if os.path.exists(vis_dir):
                logger.info(f"Recherche de visualisations dans {vis_dir}")
                for file in sorted(os.listdir(vis_dir)):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.svg')):
                        file_path = os.path.join(vis_dir, file)
                        try:
                            with open(file_path, 'rb') as img_file:
                                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                            # Vérifier la taille de l'image pour le débogage
                            file_size = os.path.getsize(file_path)
                            logger.info(f"  -> Visualisation trouvée: {file}, taille: {file_size} octets")
                            
                            # Chercher les données associées
                            fig_name = os.path.splitext(file)[0]
                            data_path = os.path.join(vis_dir, f"{fig_name}_data.json")
                            chart_data = None
                            if os.path.exists(data_path):
                                try:
                                    with open(data_path, 'r', encoding='utf-8') as f:
                                        chart_data = json.load(f)
                                    logger.info(f"  -> Données associées trouvées pour: {file}")
                                except Exception as e:
                                    logger.error(f"Impossible de lire les données associées à {file}: {e}")
                            
                            # Chercher les données CSV associées
                            csv_path = os.path.join(vis_dir, f"{fig_name}_data.csv")
                            csv_data = ""
                            if os.path.exists(csv_path):
                                try:
                                    with open(csv_path, 'r', encoding='utf-8') as f:
                                        csv_data = f.read()
                                    logger.info(f"  -> Données CSV trouvées pour: {file}")
                                except Exception as e:
                                    logger.error(f"Impossible de lire les données CSV associées à {file}: {e}")
                            
                            # Si pas de fichier CSV, essayer d'extraire des données CSV imbriquées
                            if not csv_data and chart_data:
                                # Parcourir chart_data pour trouver csv_data
                                for axes_key, axes_value in chart_data.items():
                                    if isinstance(axes_value, dict):
                                        for line_key, line_value in axes_value.items():
                                            if isinstance(line_value, dict) and 'csv_data' in line_value:
                                                csv_data = line_value['csv_data']
                                                logger.info(f"  -> Données CSV intégrées trouvées pour: {file}")
                                                break
                                    if csv_data:
                                        break
                            
                            vis_files.append({
                                'filename': file,
                                'path': file_path,
                                'base64': img_data,
                                'title': chart_data.get('title', ' '.join(os.path.splitext(file)[0].split('_')).capitalize()) if chart_data else ' '.join(os.path.splitext(file)[0].split('_')).capitalize(),
                                'size': file_size,
                                'data': chart_data,
                                'csv_data': csv_data  # Conserver les données CSV pour compatibilité
                            })
                        except Exception as e:
                            logger.error(f"Impossible de lire ou encoder l'image {file_path}: {e}")
            else:
                logger.warning(f"Répertoire de visualisations non trouvé: {vis_dir}")

            # Capturer les tables de régression
            regression_outputs = capture_regression_outputs(result.stdout, tables_dir)
            
            # Vérifier également si des tables ont été sauvegardées directement par le script
            if os.path.exists(tables_dir):
                logger.info(f"Recherche de tables de régression dans {tables_dir}")
                for file in sorted(os.listdir(tables_dir)):
                    # Ne traiter que les fichiers texte qui ne sont pas déjà traités
                    if file.lower().endswith('.txt') and not any(r['text_path'].endswith(file) for r in regression_outputs):
                        text_path = os.path.join(tables_dir, file)
                        img_path = os.path.join(tables_dir, os.path.splitext(file)[0] + '.png')
                        
                        # Lire le contenu du fichier texte
                        try:
                            with open(text_path, 'r', encoding='utf-8') as f:
                                table_content = f.read()
                            
                            # Vérifier si c'est une table de régression OLS
                            if "OLS Regression Results" in table_content:
                                table_id = os.path.splitext(file)[0]
                                
                                # Créer l'image si elle n'existe pas
                                if not os.path.exists(img_path):
                                    plt.figure(figsize=(12, 10))
                                    plt.text(0.01, 0.99, table_content, family='monospace', fontsize=10, verticalalignment='top')
                                    plt.axis('off')
                                    plt.tight_layout()
                                    plt.savefig(img_path, dpi=100, bbox_inches='tight')
                                    plt.close()
                                
                                # Encoder l'image en base64
                                with open(img_path, 'rb') as img_file:
                                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                                
                                # Extraire des métadonnées basiques
                                r_squared_match = re.search(r"R-squared:\s*([\d\.]+)", table_content)
                                r_squared = r_squared_match.group(1) if r_squared_match else "N/A"
                                
                                # Chercher le fichier de données JSON associé
                                data_path = os.path.join(tables_dir, f"{table_id}_data.json")
                                regression_data = None
                                if os.path.exists(data_path):
                                    try:
                                        with open(data_path, 'r', encoding='utf-8') as f:
                                            regression_data = json.load(f)
                                        logger.info(f"  -> Données de régression trouvées pour: {file}")
                                    except Exception as e:
                                        logger.error(f"Impossible de lire les données de régression pour {file}: {e}")
                                
                                # Chercher les données CSV des coefficients
                                csv_path = os.path.join(tables_dir, f"{table_id}_coefficients.csv")
                                csv_data = ""
                                if os.path.exists(csv_path):
                                    try:
                                        with open(csv_path, 'r', encoding='utf-8') as f:
                                            csv_data = f.read()
                                        logger.info(f"  -> Données CSV des coefficients trouvées pour: {file}")
                                    except Exception as e:
                                        logger.error(f"Impossible de lire les données CSV pour {file}: {e}")
                                
                                # Si pas de fichier CSV mais données JSON disponibles
                                if not csv_data and regression_data and "csv_data" in regression_data:
                                    csv_data = regression_data["csv_data"]
                                    logger.info(f"  -> Données CSV intégrées trouvées pour: {file}")
                                
                                # Vérifier la taille de l'image pour le débogage
                                file_size = os.path.getsize(img_path)
                                logger.info(f"  -> Table de régression trouvée: {file}, taille: {file_size} octets")
                                
                                regression_outputs.append({
                                    'type': 'regression_table',
                                    'id': table_id,
                                    'text_path': text_path,
                                    'image_path': img_path,
                                    'base64': img_data,
                                    'title': f"Résultats de Régression: {table_id.replace('_', ' ').capitalize()}",
                                    'metadata': {
                                        'r_squared': r_squared
                                    },
                                    'size': file_size,
                                    'data': regression_data,
                                    'csv_data': csv_data  # Conserver les données CSV pour compatibilité
                                })
                        except Exception as e:
                            logger.error(f"Erreur lors du traitement de la table {text_path}: {e}")

            # Mettre à jour les résultats
            execution_results["visualisations"] = vis_files
            execution_results["regressions"] = regression_outputs
            logger.info(f"{len(vis_files)} visualisation(s) et {len(regression_outputs)} table(s) de régression traitée(s).")
            
            # Journaliser plus de détails sur les visualisations pour le débogage
            for i, vis in enumerate(vis_files):
                logger.info(f"  Visualisation #{i+1}: {vis.get('filename')}, taille: {vis.get('size', 'inconnue')} octets")
                if 'base64' in vis:
                    logger.info(f"    Longueur base64: {len(vis['base64'])} caractères")
                if 'data' in vis and vis['data']:
                    logger.info(f"    Données présentes: Oui")
                if 'csv_data' in vis and vis['csv_data']:
                    logger.info(f"    Données CSV présentes: {len(vis['csv_data'])} caractères")
                
            for i, reg in enumerate(regression_outputs):
                logger.info(f"  Régression #{i+1}: {reg.get('id')}, taille: {reg.get('size', 'inconnue')} octets")
                if 'base64' in reg:
                    logger.info(f"    Longueur base64: {len(reg['base64'])} caractères")
                if 'data' in reg and reg['data']:
                    logger.info(f"    Données structurées: Oui")
                if 'csv_data' in reg and reg['csv_data']:
                    logger.info(f"    Données CSV présentes: {len(reg['csv_data'])} caractères")
            
            break

        # Gestion des erreurs d'exécution
        except subprocess.TimeoutExpired:
             logger.error("L'exécution a dépassé le délai de 300 secondes.")
             error_message = "TimeoutExpired: Le script a mis trop de temps à s'exécuter."
             recent_errors.append(error_message.strip())
             current_stderr = error_message

        except subprocess.CalledProcessError as e:
            stderr_output = e.stderr.strip()
            logger.error(f"Erreur lors de l'exécution (stderr):\n{stderr_output}")
            recent_errors.append(stderr_output)
            current_stderr = e.stderr

            # Sauvegarder l'erreur dans un fichier
            error_file = os.path.join(code_versions_dir, f"attempt_{attempt}_error.txt")
            try:
                with open(error_file, "w", encoding="utf-8") as f:
                    f.write(f"ERREUR RENCONTRÉE LORS DE LA TENTATIVE {attempt}:\n\n")
                    f.write(current_stderr)
                logger.info(f"Erreur sauvegardée dans: {error_file}")
            except Exception as write_err:
                 logger.error(f"Impossible de sauvegarder le fichier d'erreur {error_file}: {write_err}")

            # MODIFICATION: Utiliser le modèle puissant après 5 tentatives échouées
            if attempt >= 10 and not tried_powerful_model and backend == "gemini":
                logger.warning(f"Après {attempt} tentatives échouées, basculement vers le modèle Gemini Pro puissant")
                tried_powerful_model = True
                powerful_model = "gemini-2.5-pro-exp-03-25"
                current_model = powerful_model  # Mise à jour du modèle courant
                
                # Créer un prompt spécifique pour résoudre les problèmes de formule OLS
                powerful_prompt = f"""Voici mon script Python qui génère une erreur. Corrige-le avec soin:

```python
{code}
```

Erreur:
```
{current_stderr}
```

Je veux uniquement le code Python corrigé, sans explications. Le script est utilisé pour analyser des données avec pandas, matplotlib et statsmodels.

PROBLÈME À RÉSOUDRE:
- Il y a une erreur de "unterminated string literal" dans une formule OLS
- Le problème est lié aux noms de pays avec espaces et caractères spéciaux dans la formule
- La solution est d'utiliser des backticks (`) pour encadrer chaque nom de variable avec espaces/caractères spéciaux
- Exemple: `Pays_Afrique du Sud` au lieu de Pays_Afrique du Sud
- OU simplifier la formule en utilisant une technique d'encodage différente pour les pays
"""
                
                # Sauvegarder le prompt pour le modèle puissant
                save_prompt_to_file(powerful_prompt, prompts_log_path, "Powerful Model after 5 attempts")
                logger.info(f"Envoi du prompt de correction au modèle puissant {powerful_model}")
                
                try:
                    powerful_correction = call_llm(prompt=powerful_prompt, model_name=powerful_model, backend="gemini")
                    corrected_code = extract_code(powerful_correction)
                    if corrected_code and len(corrected_code.strip()) > 0:
                        code = corrected_code
                        logger.info("Code corrigé par le modèle puissant reçu")
                        recent_errors.clear()
                        continue
                    else:
                        logger.warning("Le modèle puissant n'a pas renvoyé de code utilisable")
                except Exception as powerful_err:
                    logger.error(f"Erreur avec le modèle puissant: {powerful_err}")

            # Vérifier si les erreurs se répètent
            if len(recent_errors) == 3 and len(set(recent_errors)) == 1:
                logger.warning("Trois erreurs identiques consécutives détectées.")
                
                # Si on n'a pas encore essayé le modèle puissant après 3 erreurs identiques
                if not tried_powerful_model and backend == "gemini":
                    logger.warning("Basculement vers le modèle Gemini Pro puissant pour correction")
                    tried_powerful_model = True
                    powerful_model = "gemini-2.5-pro-exp-03-25"
                    current_model = powerful_model  # Mise à jour du modèle courant
                    
                    # Créer un prompt simplifié
                    powerful_prompt = f"""Voici mon script Python qui génère une erreur. Corrige-le avec soin:

```python
{code}
```

Erreur:
```
{current_stderr}
```

Je veux uniquement le code Python corrigé, sans explications. Le script est utilisé pour analyser des données avec pandas et matplotlib.
"""
                    
                    # Sauvegarder le prompt pour le modèle puissant
                    save_prompt_to_file(powerful_prompt, prompts_log_path, "Powerful Model Correction")
                    logger.info(f"Envoi du prompt de correction au modèle puissant {powerful_model}")
                    
                    try:
                        powerful_correction = call_llm(prompt=powerful_prompt, model_name=powerful_model, backend="gemini")
                        corrected_code = extract_code(powerful_correction)
                        if corrected_code and len(corrected_code.strip()) > 0:
                            code = corrected_code
                            logger.info("Code corrigé par le modèle puissant reçu")
                            recent_errors.clear()
                            continue
                        else:
                            logger.warning("Le modèle puissant n'a pas renvoyé de code utilisable")
                    except Exception as powerful_err:
                        logger.error(f"Erreur avec le modèle puissant: {powerful_err}")
                
                # Si le modèle puissant a échoué ou a déjà été essayé, passer à la correction manuelle
                logger.warning("Passage à la correction manuelle.")
                manual_edit_path = os.path.join(code_versions_dir, f"manual_fix_for_attempt_{attempt+1}.py")

                if not os.path.exists(manual_edit_path):
                    try:
                        with open(manual_edit_path, "w", encoding="utf-8") as f:
                            f.write(code)
                        logger.info(f"Fichier de correction manuelle créé : {manual_edit_path}")
                    except Exception as create_err:
                        logger.error(f"Erreur lors de la création du fichier de correction manuelle {manual_edit_path}: {create_err}")
                else:
                    try:
                        import shutil
                        shutil.copyfile(manual_edit_path, manual_edit_path + ".bak")
                        logger.info(f"Backup du fichier de correction manuelle créé : {manual_edit_path}.bak")
                    except Exception as copy_err:
                        logger.error(f"Erreur lors de la sauvegarde du fichier {manual_edit_path}: {copy_err}")

                logger.info(f"Modifiez le fichier {manual_edit_path} si nécessaire.")

                import sys

                # Afficher le message sur stderr (pas stdout)
                sys.stderr.write("⏸️ Appuyez sur Entrée pour continuer après modification...\n")
                sys.stderr.flush()  # Assurez-vous que le message s'affiche immédiatement
                try:
                    input()  # Pas de message dans input() pour éviter qu'il aille dans stdout
                except EOFError:
                    logger.warning("Pas d'entrée utilisateur détectée, poursuite automatique.")

                try:
                    with open(manual_edit_path, "r", encoding="utf-8") as f:
                        edited_code = f.read()
                except Exception as read_err:
                    logger.error(f"Erreur lors de la lecture du fichier {manual_edit_path}: {read_err}")
                    edited_code = code

                if edited_code != code:
                    logger.info(f"Code modifié détecté dans {manual_edit_path}. Mise à jour pour la prochaine tentative.")
                    code = edited_code
                    recent_errors.clear()
                    continue
                else:
                    logger.info(f"Aucune modification détectée dans {manual_edit_path}. Passage à la correction automatique par LLM.")

            # Tentative de correction automatique par LLM standard
            if attempt < max_attempts:
                llm_correction_attempt += 1
                logger.info(f"Tentative de correction LLM #{llm_correction_attempt}")
                
                # Créer le prompt pour la correction
                metadata_str = json.dumps(agent1_data["metadata"], indent=2, ensure_ascii=False)
                recall_prompt = f"""
Le contexte suivant concerne l'analyse d'un fichier CSV :
{metadata_str}

Le chemin absolu du fichier CSV est : {csv_file}
Assure-toi d'utiliser ce chemin exact dans pd.read_csv('{csv_file}').

Le code Python ci-dessous, complet avec toutes ses fonctionnalités (gestion des visualisations, sauvegarde des versions, correction automatique et manuelle, etc.), a généré l'erreur suivante lors de son exécution :
------------------------------------------------------------
Code Fautif :
```python
{code}
```
Erreur Rencontrée : {current_stderr}
TA MISSION : Corrige uniquement l'erreur indiquée sans modifier la logique globale du code. Garde intégralement la structure et l'ensemble des fonctionnalités du code initial. Ne simplifie pas le script : toutes les parties (gestion des visualisations, sauvegarde des versions, correction manuelle, etc.) doivent être conservées. GARDE les noms de colonnes exacts. Colonnes valides : {agent1_data["metadata"].get("noms_colonnes", [])}

RENVOIE UNIQUEMENT le code Python corrigé, encapsulé dans un bloc de code délimité par trois backticks (python ... ), sans explications supplémentaires. 

Fais bien attention a la nature des variables, numériques, catégorielles, etc.

IMPORTANT: Pour les styles dans matplotlib, utilise 'seaborn-v0_8-whitegrid' au lieu de 'seaborn-whitegrid' qui est obsolète.
"""

                # Sauvegarder le prompt
                save_prompt_to_file(recall_prompt, prompts_log_path, f"Code Correction Attempt {attempt} (LLM #{llm_correction_attempt})")
                logger.info(f"Envoi du prompt de correction au LLM via backend '{backend}' avec modèle '{current_model}'...")
                
                try:
                    # Obtenir et traiter la correction du LLM
                    new_generated_script = call_llm(prompt=recall_prompt, model_name=current_model, backend=backend)
                    
                    extracted_code = extract_code(new_generated_script)
                    clean_code = remove_shell_commands(extracted_code)

                    if not clean_code.strip():
                        logger.warning("Le LLM a renvoyé un code vide après nettoyage. Réutilisation du code précédent.")
                    else:
                        code = clean_code # Mettre à jour le code pour la prochaine tentative
                        logger.info("Code corrigé par le LLM reçu et nettoyé.")

                    time.sleep(1)

                except Exception as llm_call_err:
                    logger.error(f"Erreur lors de l'appel au LLM pour correction: {llm_call_err}")
                    
                    # Si l'appel au LLM standard a échoué et qu'on n'a pas encore essayé le modèle puissant
                    if not tried_powerful_model and backend == "gemini":
                        logger.warning("Tentative avec le modèle Gemini Pro puissant après échec du LLM standard")
                        tried_powerful_model = True
                        powerful_model = "gemini-2.5-pro-exp-03-25"
                        current_model = powerful_model  # Mise à jour du modèle courant
                        
                        # Créer un prompt simplifié
                        powerful_prompt = f"""Voici mon script Python qui génère une erreur. Corrige-le avec soin:

```python
{code}
```

Erreur:
```
{current_stderr}
```

Je veux uniquement le code Python corrigé, sans explications. Le script est utilisé pour analyser des données avec pandas et matplotlib. 
Assure-toi d'utiliser 'seaborn-v0_8-whitegrid' au lieu de 'seaborn-whitegrid' qui est obsolète.
"""
                        
                        # Sauvegarder le prompt pour le modèle puissant
                        save_prompt_to_file(powerful_prompt, prompts_log_path, "Powerful Model Correction After LLM Failure")
                        logger.info(f"Envoi du prompt de correction au modèle puissant {powerful_model}")
                        
                        try:
                            powerful_correction = call_llm(prompt=powerful_prompt, model_name=powerful_model, backend="gemini")
                            corrected_code = extract_code(powerful_correction)
                            if corrected_code and len(corrected_code.strip()) > 0:
                                code = corrected_code
                                logger.info("Code corrigé par le modèle puissant reçu")
                                recent_errors.clear()
                                continue
                            else:
                                logger.warning("Le modèle puissant n'a pas renvoyé de code utilisable")
                        except Exception as powerful_err:
                            logger.error(f"Erreur avec le modèle puissant: {powerful_err}")
                    
                    # Si aucun LLM ne fonctionne, passer à la correction manuelle en dernier recours
                    logger.warning("Toutes les tentatives de correction automatique ont échoué. Passage à la correction manuelle.")
                    manual_edit_path = os.path.join(code_versions_dir, f"manual_fix_final_attempt.py")
                    try:
                        with open(manual_edit_path, "w", encoding="utf-8") as f:
                            f.write(code)
                        logger.info(f"Fichier de correction manuelle finale créé : {manual_edit_path}")
                        
                        # Attendre la correction manuelle
                        sys.stderr.write("⏸️ CORRECTION FINALE MANUELLE REQUISE. Modifiez le fichier puis appuyez sur Entrée...\n")
                        sys.stderr.flush()
                        try:
                            input()
                        except EOFError:
                            logger.warning("Pas d'entrée utilisateur détectée, poursuite automatique.")
                            
                        with open(manual_edit_path, "r", encoding="utf-8") as f:
                            edited_code = f.read()
                        code = edited_code
                        recent_errors.clear()
                        
                    except Exception as manual_err:
                        logger.error(f"Erreur lors de la correction manuelle finale: {manual_err}")
                        execution_results = {
                            "success": False, 
                            "error": f"Échec complet - correction automatique et manuelle: {manual_err}",
                            "stderr": current_stderr,
                            "all_code_versions": all_code_versions
                        }
                        break

        except Exception as general_err:
            logger.error(f"Erreur générale inattendue lors de la tentative {attempt}: {general_err}", exc_info=True)
            error_message = f"Erreur générale inattendue: {general_err}"
            recent_errors.append(error_message.strip())
            execution_results = {
                "success": False,
                "error": error_message,
                "stderr": current_stderr if current_stderr else str(general_err),
                "all_code_versions": all_code_versions
            }
            break

    # Ajouter les interprétations pour les visualisations et tables de régression
    if execution_results.get("success"):
        all_visuals = []
        
        # Traiter les visualisations
        if "visualisations" in execution_results:
            all_visuals.extend(execution_results["visualisations"])
            
        # Traiter les tables de régression
        if "regressions" in execution_results:
            all_visuals.extend(execution_results["regressions"])
        
        if all_visuals:
            logger.info("Génération d'interprétations pour les visualisations et tables avec Gemini")
            all_visuals_with_interpretations = generate_visualization_interpretations(
                execution_results.get("visualisations", []),
                execution_results.get("regressions", []),
                agent1_data,
                current_model,  # Utilisation du modèle courant (qui peut être le modèle puissant)
                backend,
                prompts_log_path
            )
            
            # Mettre à jour les résultats avec les interprétations
            execution_results["all_visuals"] = all_visuals_with_interpretations

    # Finaliser et journaliser les résultats
    final_status = "success" if execution_results.get("success") else "failed"
    logger.info(f"Fin de la boucle d'exécution. Statut final: {final_status}. Total tentatives: {attempt}")

    # Sauvegarder une copie du code final
    final_script_path = None
    last_code_version_path = all_code_versions[-1] if all_code_versions else None

    if last_code_version_path and os.path.exists(last_code_version_path):
        final_script_name = f"analysis_script_{timestamp}_{final_status}.py"
        outputs_dir = "outputs"
        os.makedirs(outputs_dir, exist_ok=True)
        final_script_path = os.path.join(outputs_dir, final_script_name)

        try:
            shutil.copyfile(last_code_version_path, final_script_path)
            logger.info(f"Version finale du code ({final_status}) copiée vers: {final_script_path}")
        except Exception as copy_err:
            logger.error(f"Impossible de copier le script final {last_code_version_path} vers {final_script_path}: {copy_err}")
            final_script_path = None
    elif not all_code_versions:
        logger.warning("Aucune version de code n'a été générée ou sauvegardée.")
    else:
        logger.warning(f"Le dernier fichier de code {last_code_version_path} n'existe pas. Impossible de sauvegarder le script final.")

    # Créer un fichier d'index pour documenter la session
    index_file_path = os.path.join(code_versions_dir, "index.txt")
    try:
        with open(index_file_path, "w", encoding="utf-8") as f:
            f.write(f"SESSION D'ANALYSE DU {timestamp}\n")
            f.write(f"Fichier CSV: {csv_file}\n")
            f.write(f"Modèle LLM: {model}\n")
            f.write(f"Modèle LLM courant à la fin: {current_model}\n")
            f.write(f"Nombre de versions de code générées/sauvegardées: {len(all_code_versions)}\n")
            f.write(f"Nombre total de tentatives d'exécution effectuées: {attempt}\n")
            f.write(f"Nombre de tentatives de correction LLM: {llm_correction_attempt}\n")
            f.write(f"Modèle puissant utilisé: {'Oui' if tried_powerful_model else 'Non'}\n")
            f.write(f"Résultat final: {'Succès' if execution_results.get('success') else 'Échec'}\n")

            if not execution_results.get("success"):
                f.write(f"Dernière erreur enregistrée: {execution_results.get('error', 'N/A')}\n")
                last_stderr = execution_results.get('stderr')
                if last_stderr:
                    f.write(f"Dernier stderr:\n```\n{last_stderr}\n```\n")

            f.write("\nLISTE DES VERSIONS DE CODE GÉNÉRÉES/UTILISÉES:\n")
            for i, path in enumerate(all_code_versions, 1):
                f.write(f"- Version {i} (Tentative {i}): {os.path.basename(path)}\n")
                error_path = path.replace(".py", "_error.txt")
                if os.path.exists(error_path):
                    f.write(f"  - Erreur associée: {os.path.basename(error_path)}\n")
                manual_edit_trigger_path = os.path.join(os.path.dirname(path), f"manual_fix_for_attempt_{i+1}.py")
                if os.path.exists(manual_edit_trigger_path):
                    f.write(f"  - Fichier pour édition manuelle (créé après échec tentative {i}): {os.path.basename(manual_edit_trigger_path)}\n")

            if final_script_path and os.path.exists(final_script_path):
                f.write(f"\nScript final utilisé ({final_status}): {os.path.basename(final_script_path)}\n")
                f.write(f"Chemin complet: {final_script_path}\n")
            else:
                if last_code_version_path and not os.path.exists(last_code_version_path):
                    f.write(f"\nScript final non sauvegardé (fichier source {os.path.basename(last_code_version_path)} introuvable).\n")
                elif final_script_path is None and last_code_version_path:
                    f.write(f"\nScript final non sauvegardé (échec de la copie depuis {os.path.basename(last_code_version_path)}).\n")
                else:
                    f.write("\nScript final non sauvegardé (aucune version de code disponible).\n")

            if os.path.exists(prompts_log_path):
                f.write(f"\nJournal des prompts envoyés au LLM: {os.path.basename(prompts_log_path)}\n")
                f.write(f"Chemin complet: {prompts_log_path}\n")

            f.write(f"\nRépertoire des versions de code: {code_versions_dir}\n")
            if execution_results.get("success"):
                vis_dir_final = execution_results.get('vis_dir', os.path.join(temp_dir, "visualisations"))
                tables_dir_final = execution_results.get('tables_dir', os.path.join(temp_dir, "tables"))
                f.write(f"Répertoire des visualisations: {vis_dir_final}\n")
                f.write(f"Répertoire des tables de régression: {tables_dir_final}\n")
            temp_dir_final = execution_results.get('temp_dir', temp_dir)
            f.write(f"Répertoire temporaire d'exécution: {temp_dir_final}\n")
        logger.info(f"Fichier d'index créé: {index_file_path}")
    except Exception as index_err:
        logger.error(f"Impossible de créer le fichier d'index {index_file_path}: {index_err}")
        index_file_path = None

# Ajouter les chemins finaux aux résultats retournés
    execution_results["final_script_path"] = final_script_path
    execution_results["code_versions_dir"] = code_versions_dir
    execution_results["index_file"] = index_file_path
    execution_results["current_model"] = current_model  # Ajouter le modèle utilisé pour les dernières interprétations

    # Traiter les tables de régression avec une interprétation dédiée
    regression_outputs = execution_results.get("regressions", [])
    if regression_outputs:
        logger.info(f"Génération d'interprétations détaillées pour {len(regression_outputs)} tables de régression")
        
        for reg in regression_outputs:
            if 'data' in reg:
                # Récupérer le code de régression
                regression_code = reg.get('regression_code', '')
                
                # Appel à notre nouvelle fonction pour interpréter la régression
                detailed_interpretation = interpret_regression_with_llm(
                    reg['data'],
                    regression_code,
                    agent1_data,
                    current_model,  # Utiliser le modèle courant qui peut être le modèle puissant
                    backend,
                    prompts_log_path,
                    timeout=180  # Plus de temps pour l'analyse détaillée
                )
                
                # Ajouter l'interprétation détaillée au résultat
                reg['detailed_interpretation'] = detailed_interpretation
                logger.info(f"Interprétation détaillée générée pour la régression {reg.get('id', 'inconnue')}")
            else:
                logger.warning(f"Pas de données structurées pour la régression {reg.get('id', 'inconnue')}. Interprétation détaillée impossible.")
    
    # Mettre à jour les résultats avec les régressions interprétées
    execution_results["regressions"] = regression_outputs

    return execution_results

# Nouvelle fonction à ajouter dans agent2.py

def interpret_regression_with_llm(regression_data, code_executed, agent1_data, model, backend, prompts_log_path, timeout=120):
    """
    Interprète de manière détaillée les résultats d'une régression économétrique en utilisant un LLM.
    
    Args:
        regression_data: Données structurées de la régression (coefficients, r-squared, etc.)
        code_executed: Code Python qui a généré cette régression
        agent1_data: Données de l'agent1 pour le contexte
        model: Modèle LLM à utiliser
        backend: Backend pour les appels LLM
        prompts_log_path: Chemin pour sauvegarder les prompts
        timeout: Timeout en secondes. Défaut: 120.
        
    Returns:
        str: Interprétation détaillée des résultats de régression
    """
    from llm_utils import call_llm
    from datetime import datetime
    
    # Extraire les informations clés de la régression
    r_squared = regression_data.get('r_squared', 'N/A')
    coefficients = regression_data.get('coefficients', [])
    
    # Préparer un résumé des coefficients pour le prompt
    coef_summary = ""
    for coef in coefficients:
        var_name = coef.get('variable', 'Inconnu')
        coef_value = coef.get('coef', 'N/A')
        p_value = coef.get('p_value', 'N/A')
        std_err = coef.get('std_err', 'N/A')
        coef_summary += f"{var_name}: coefficient={coef_value}, p-value={p_value}, std_err={std_err}\n"
    
    # Extraire le contexte de recherche de l'agent1
    user_prompt = agent1_data.get('user_prompt', 'Non disponible')
    introduction = agent1_data.get('llm_output', {}).get('introduction', 'Non disponible')
    hypotheses = agent1_data.get('llm_output', {}).get('hypotheses', 'Non disponible')
    
    # Extraire les noms des colonnes pour avoir une idée des variables disponibles
    column_names = agent1_data.get('metadata', {}).get('noms_colonnes', [])
    
    # Créer le prompt pour l'interprétation de la régression
    prompt = f"""## INTERPRÉTATION ÉCONOMÉTRIQUE DÉTAILLÉE

### Résultats de régression
R-squared: {r_squared}

### Coefficients
{coef_summary}

### Code Python ayant généré cette régression
```python
{code_executed}
```

### Question de recherche
{user_prompt}

### Contexte académique
{introduction[:500]}...

### Hypothèses de recherche
{hypotheses}

### Variables disponibles dans le dataset
{', '.join(column_names)}

---

En tant qu'économètre expert, ton objectif est de fournir une interprétation précise, rigoureuse et académique de ces résultats de régression.

Ton interprétation doit inclure:

1. **Analyse du modèle global**
   - Qualité globale de l'ajustement (R²)
   - Validité statistique du modèle
   - Adéquation du modèle à la question de recherche

2. **Interprétation des coefficients significatifs**
   - Analyse détaillée de chaque coefficient statistiquement significatif (p < 0.05)
   - Interprétation précise de l'effet marginal (magnitude et direction)
   - Unités de mesure et contexte économique de chaque effet

3. **Implications économiques**
   - Mécanismes économiques sous-jacents expliquant les relations observées
   - Liens avec les théories économiques pertinentes
   - Implications pour la question de recherche initiale

4. **Limites de l'estimation**
   - Problèmes potentiels d'endogénéité, de variables omises ou de causalité
   - Robustesse des résultats
   - Pistes d'amélioration du modèle

IMPORTANT:
- Base ton analyse uniquement sur les résultats statistiques fournis, tout en les contextualisant avec la question de recherche
- Utilise un langage économétrique précis (élasticités, effets marginaux, significativité, etc.)
- Procède coefficient par coefficient pour les variables significatives
- Fais le lien entre les résultats statistiques et les mécanismes économiques
- Ton analyse doit être concise mais complète, en 3-4 paragraphes

Il est essentiel que cette interprétation soit suffisamment détaillée pour former le cœur d'une analyse économétrique académique rigoureuse.
"""

    # Sauvegarder le prompt dans le fichier journal
    try:
        with open(prompts_log_path, "a", encoding="utf-8") as f:
            f.write("\n\n" + "="*80 + "\n")
            f.write(f"PROMPT POUR INTERPRÉTATION DE RÉGRESSION - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
            f.write(prompt)
            f.write("\n" + "="*80 + "\n")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du prompt: {e}")
    
    try:
        logger.info(f"Appel au LLM pour l'interprétation détaillée de la régression (R²={r_squared})")
        interpretation = call_llm(
            prompt=prompt, 
            model_name=model, 
            backend=backend,
            timeout=timeout
        )
        
        logger.info("Interprétation détaillée de la régression générée avec succès")
        return interpretation
    except Exception as e:
        logger.error(f"Erreur lors de l'interprétation détaillée de la régression: {e}")
        return f"Erreur lors de l'interprétation détaillée de la régression: {e}"

# Fonction pour extraire le code qui a généré une régression
def extract_regression_code(full_code, table_content):
    """Tente d'identifier le code qui a généré une régression spécifique."""
    if not full_code:
        return ""
        
    # Rechercher des patterns communs dans les codes de régression
    import re
    
    # Extraire les noms de variables potentiels de la table de régression
    var_names = re.findall(r"([A-Za-z0-9_]+)\s+[\d\.-]+\s+[\d\.-]+\s+[\d\.-]+\s+[\d\.-]+", table_content)
    
    # Chercher les modèles OLS qui utilisent ces variables
    regression_patterns = [
        r"(.*?sm\.OLS.*?\.fit\(\).*?)(?=\n\n|\Z)",  # Pattern pour statsmodels OLS
        r"(.*?statsmodels\.api.*?OLS.*?\.fit\(\).*?)(?=\n\n|\Z)",  # Autre pattern pour statsmodels
        r"(.*?LinearRegression\(\).*?\.fit\(.*?\).*?)(?=\n\n|\Z)"   # Pattern pour sklearn LinearRegression
    ]
    
    # Si nous avons des noms de variables, chercher des lignes qui les utilisent
    if var_names:
        var_patterns = []
        for var in var_names[:5]:  # Limiter aux 5 premières variables pour éviter de surcharger
            var_patterns.append(r".*?" + re.escape(var) + r".*?\n")
        
        # Chercher des blocs de code qui contiennent à la fois un modèle de régression
        # et les variables identifiées
        for reg_pattern in regression_patterns:
            reg_blocks = re.findall(reg_pattern, full_code, re.DOTALL)
            for block in reg_blocks:
                # Vérifier si ce bloc contient au moins une des variables
                for var_pattern in var_patterns:
                    if re.search(var_pattern, block, re.MULTILINE):
                        # Bloc de code qui contient à la fois un modèle de régression et une variable pertinente
                        return block.strip()
    
    # Si nous n'avons pas trouvé de code spécifique, chercher simplement un modèle OLS
    for pattern in regression_patterns:
        matches = re.findall(pattern, full_code, re.DOTALL)
        if matches:
            # Prendre le premier bloc de code qui correspond à un modèle de régression
            return matches[0].strip()
    
    # Si nous n'avons toujours rien trouvé, retourner une portion du code global
    # qui contient des mots-clés de régression
    regression_keywords = ["OLS", "regression", "statsmodels", "LinearRegression", "fit("]
    lines = full_code.split('\n')
    for i, line in enumerate(lines):
        for keyword in regression_keywords:
            if keyword in line:
                # Retourner un bloc de 10 lignes autour de cette ligne si possible
                start = max(0, i - 5)
                end = min(len(lines), i + 6)
                return '\n'.join(lines[start:end])
    
    # Si tout échoue, retourner une chaîne vide
    return ""

def generate_analysis_code(csv_file, user_prompt, agent1_data, model, prompts_log_path, backend: str):
    """
    Génère le code d'analyse économétrique de niveau accessible
    
    Args:
        csv_file: Chemin absolu du fichier CSV
        user_prompt: Prompt initial de l'utilisateur
        agent1_data: Données de l'agent1
        model: Modèle LLM à utiliser
        prompts_log_path: Chemin pour sauvegarder les prompts
        backend: Backend pour les appels LLM
        
    Returns:
        str: Code d'analyse généré
    """
    # Préparation du prompt pour le LLM avec le chemin absolu du fichier et les noms de colonnes exacts
    metadata_str = json.dumps(agent1_data["metadata"], indent=2, ensure_ascii=False)
    colonnes = agent1_data["metadata"].get("noms_colonnes", [])
    col_dict = ",\n".join([f'    "{col}": "{col}"' for col in colonnes])
    
    # Extraire les sections si disponibles
    introduction = agent1_data['llm_output'].get('introduction', 'Non disponible')
    literature_review = agent1_data['llm_output'].get('literature_review', 'Non disponible')
    hypotheses = agent1_data['llm_output'].get('hypotheses', 'Non disponible')
    methodology = agent1_data['llm_output'].get('methodology', 'Non disponible')
    limitations = agent1_data['llm_output'].get('limitations', 'Non disponible')
    variables_info = agent1_data['llm_output'].get('variables', 'Non disponible')
    
# Utiliser les anciennes sections si les nouvelles ne sont pas disponibles
    if introduction == 'Non disponible':
        introduction = agent1_data['llm_output'].get('problematisation', 'Non disponible')
    
    if methodology == 'Non disponible':
        methodology = agent1_data['llm_output'].get('approches_suggerees', 'Non disponible')
    
    if limitations == 'Non disponible':
        limitations = agent1_data['llm_output'].get('points_vigilance', 'Non disponible')
    
    # Créer le prompt pour la génération de code
    prompt = f"""## GÉNÉRATION DE CODE D'ANALYSE DE DONNÉES

### Fichier CSV et Métadonnées
```json
{metadata_str}
```

### Chemin absolu du fichier CSV
{csv_file}

### Noms exacts des colonnes à utiliser
{colonnes}

### Introduction et problématique de recherche
{introduction}

### Hypothèses de recherche
{hypotheses}

### Méthodologie proposée
{methodology}

### Limites identifiées
{limitations}

### Informations sur les variables
{variables_info}

### Demande initiale de l'utilisateur
{user_prompt}

---

Tu es un analyste de données expérimenté. Ta mission est de générer un script Python d'analyse de données clair et accessible. Le code doit être robuste et produire des visualisations attrayantes.

DIRECTIVES:

1. CHARGEMENT ET PRÉTRAITEMENT DES DONNÉES
   - Utilise strictement le chemin absolu '{csv_file}'
   - Nettoie les données (valeurs manquantes, outliers)
   - Crée des statistiques descriptives claires

2. VISUALISATIONS ATTRAYANTES ET INFORMATIVES
   - Crée au moins 4-5 visualisations avec matplotlib/seaborn:
     * Matrice de corrélation colorée et lisible
     * Distributions des variables principales
     * Relations entre variables importantes
     * Graphiques adaptés au type de données
   - Utilise des couleurs attrayantes et des styles modernes
   - Ajoute des titres clairs, des légendes informatives et des ÉTIQUETTES D'AXES EXPLICITES
   - IMPORTANT: Assure-toi d'utiliser ax.set_xlabel() et ax.set_ylabel() avec des descriptions claires
   - IMPORTANT: Assure-toi que les graphiques soient sauvegardés ET affichés
   - Utilise plt.savefig() AVANT plt.show() pour chaque graphique
   - IMPORTANT: Pour les styles Seaborn, utilise 'seaborn-v0_8-whitegrid' au lieu de 'seaborn-whitegrid' qui est obsolète

3. MODÉLISATION SIMPLE ET CLAIRE
   - Implémente les modèles de régression appropriés
   - Utilise statsmodels avec des résultats complets
   - Présente les résultats de manière lisible
   - Documente clairement chaque étape

4. TESTS DE BASE
   - Vérifie la qualité du modèle avec des tests simples
   - Analyse les résidus
   - Vérifie la multicolinéarité si pertinent

5. CAPTURE ET STOCKAGE DES DONNÉES POUR INTERPRÉTATION
   - IMPORTANT: Pour chaque visualisation, stocke le DataFrame utilisé dans une variable
   - IMPORTANT: Après chaque création de figure, stocke les données utilisées pour permettre une interprétation précise
   - Assure-toi que chaque figure peut être associée aux données qui ont servi à la générer

EXIGENCES TECHNIQUES:
- Utilise pandas, numpy, matplotlib, seaborn, et statsmodels
- Organise ton code en sections clairement commentées
- Utilise ce dictionnaire pour accéder aux colonnes:
```python
col = {{
{col_dict}
}}
```
- Document chaque étape de façon simple et accessible
- Pour chaque visualisation:
  * UTILISE des titres clairs pour les graphiques et les axes
  * SAUVEGARDE avec plt.savefig() PUIS
  * AFFICHE avec plt.show()
- Pour les tableaux de régression, utilise print(results.summary())

IMPORTANT:
- Adapte l'analyse aux données disponibles
- Mets l'accent sur les visualisations attrayantes et bien étiquetées
- Assure-toi que chaque graphique a des étiquettes d'axe claires via ax.set_xlabel() et ax.set_ylabel()
- Assure-toi que chaque graphique est à la fois SAUVEGARDÉ et AFFICHÉ
- Utilise plt.savefig() AVANT plt.show() pour chaque graphique
- IMPORTANT: Pour les styles Seaborn, utilise 'whitegrid' au lieu de 'seaborn-whitegrid' ou 'seaborn-v0_8-whitegrid' qui sont obsolètes
"""

    # Sauvegarder le prompt dans le fichier journal
    save_prompt_to_file(prompt, prompts_log_path, "Initial Code Generation")
    logger.info("Appel LLM pour génération du code d'analyse")
    try:
        # Utilisation de llm_utils.call_llm
        generated_output = call_llm(prompt=prompt, model_name=model, backend=backend)
        
        # Extraire et nettoyer le code généré
        generated_code = extract_code(generated_output)
        clean_code = remove_shell_commands(generated_code)
        
        # S'assurer que print(results.summary()) est présent
        if "results.summary()" in clean_code and "print(results.summary())" not in clean_code:
            clean_code = clean_code.replace("results.summary()", "print(results.summary())")
        
        # Assurer que les styles Seaborn sont compatibles
        clean_code = clean_code.replace("seaborn-v0_8-whitegrid", "whitegrid")
        clean_code = clean_code.replace("seaborn-whitegrid", "whitegrid")
        clean_code = clean_code.replace("seaborn-v0_8-white", "white")
        clean_code = clean_code.replace("seaborn-white", "white")
        clean_code = clean_code.replace("seaborn-v0_8-darkgrid", "darkgrid")
        clean_code = clean_code.replace("seaborn-darkgrid", "darkgrid")
        
        return clean_code
        
    except Exception as e:
        logger.error(f"Erreur lors de l'appel LLM pour génération: {e}")
        sys.exit(1)

def generate_analysis_narrative(execution_results, agent1_data, model):
    """
    Fonction de placeholder pour la génération de narration explicative.
    Cette fonction peut être implémentée pour générer une narration basée sur les résultats.
    
    Args:
        execution_results: Résultats de l'exécution du code
        agent1_data: Données de l'agent1
        model: Modèle LLM à utiliser
        
    Returns:
        Dictionnaire avec une narration par défaut
    """
    # Pour l'instant, retourne juste un dictionnaire avec une narration par défaut
    return {
        "narrative": "Analyse exécutée avec succès. Veuillez consulter les visualisations et les résultats."
    }

def main():
    """
    Fonction principale qui exécute le pipeline d'analyse économétrique.
    """
    parser = argparse.ArgumentParser(
        description="Agent 2: Analyse économétrique"
    )
    parser.add_argument("csv_file", help="Chemin vers le fichier CSV")
    parser.add_argument("user_prompt", help="Prompt initial de l'utilisateur")
    parser.add_argument("agent1_output", help="Chemin vers le fichier de sortie de l'agent 1")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Modèle LLM à utiliser")
    parser.add_argument("--backend", default="gemini", choices=["ollama", "gemini"], help="Backend LLM à utiliser: 'ollama' ou 'gemini'")
    parser.add_argument("--auto-confirm", action="store_true", help="Ignore la pause manuelle pour correction")
    parser.add_argument("--log-file", help="Fichier de log spécifique")  # Nouvel argument
    args = parser.parse_args()

    # Reconfigurer le logging si un fichier de log spécifique est fourni
    if args.log_file:
        # Supprimer les handlers existants
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Ajouter les nouveaux handlers
        file_handler = logging.FileHandler(args.log_file, mode='a')  # mode 'a' pour append
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        stream_handler = logging.StreamHandler(sys.stderr)
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(stream_handler)
        
        logger.info(f"Logging redirigé vers {args.log_file}")

    # Créer les répertoires de sortie et les noms de fichiers
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("outputs", exist_ok=True)
    code_versions_dir = os.path.join("outputs", f"code_versions_{timestamp}")
    prompts_log_path = os.path.join(code_versions_dir, "prompts.txt")

    # Charger les données de l'agent 1
    try:
        with open(args.agent1_output, "r", encoding="utf-8") as f:
            agent1_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Erreur: Le fichier de l'agent 1 '{args.agent1_output}' n'a pas été trouvé.")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Erreur: Impossible de décoder le JSON du fichier de l'agent 1 '{args.agent1_output}'.")
        sys.exit(1)

    # Générer le code d'analyse économétrique
    logger.info("Génération du code d'analyse économétrique")
    analysis_code = generate_analysis_code(
        args.csv_file,
        args.user_prompt,
        agent1_data,
        args.model,
        prompts_log_path,
        args.backend
    )
    if not analysis_code:
         logger.error("La génération du code initial a échoué. Arrêt.")
         sys.exit(1)

    # Exécuter le code d'analyse
    logger.info("Exécution du code d'analyse")
    execution_results = attempt_execution_loop(
        analysis_code,
        args.csv_file,
        agent1_data,
        args.model,
        prompts_log_path,
        args.backend
    )

    # Générer la narration explicative des résultats
    narrative = {"narrative": "L'analyse n'a pas été exécutée avec succès ou a échoué."}
    if execution_results.get("success"):
        logger.info("Génération de la narration explicative des résultats")
        narrative = generate_analysis_narrative(
            execution_results,
            agent1_data,
            execution_results.get("current_model", args.model)  # Utiliser le modèle courant (qui peut être le puissant)
        )
    else:
        error_msg = execution_results.get('error', 'Erreur inconnue lors de l\'exécution')
        stderr_msg = execution_results.get('stderr')
        full_error = f"L'analyse a échoué: {error_msg}"
        if stderr_msg:
            full_error += f"\nDernier Stderr:\n{stderr_msg[:500]}..."
        narrative = {
            "narrative": full_error
        }
        logger.warning(f"L'exécution du code a échoué. Raison: {error_msg}")

    # Combiner toutes les visualisations et tables de régression en une seule liste
    all_visuals = execution_results.get("all_visuals", [])
    if not all_visuals:
        all_visuals = execution_results.get("visualisations", []) + execution_results.get("regressions", [])

    # Préparer la sortie finale
    output = {
        "initial_generated_code": analysis_code,
        "execution_results": execution_results,
        "narrative": narrative.get("narrative"),
        "final_script_path": execution_results.get("final_script_path", "Non disponible"),
        "visualisations": all_visuals,  # Liste combinée
        "prompts_log_file": prompts_log_path if os.path.exists(prompts_log_path) else "Non généré ou erreur",
        "output_directory": code_versions_dir,
        "index_file": execution_results.get("index_file", "Non généré ou erreur"),
        "current_model": execution_results.get("current_model", args.model)  # Ajouter le modèle utilisé pour les dernières actions
    }

    # Afficher la sortie au format JSON
    try:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    except TypeError as e:
         logger.error(f"Erreur lors de la sérialisation en JSON: {e}")
         simplified_output = {k: str(v) for k, v in output.items()}
         print(json.dumps(simplified_output, ensure_ascii=False, indent=2))

    return 0 if execution_results.get("success") else 1

if __name__ == "__main__":
    sys.exit(main())