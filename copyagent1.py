#!/usr/bin/env python3
import pandas as pd
import subprocess
import sys
import argparse
import re

def generate_metadata(csv_file):
    """
    Lit le fichier CSV et extrait les métadonnées :
    - Chemin du fichier CSV
    - Nombre de lignes
    - Nombre de colonnes
    - Noms des colonnes
    - Types des colonnes
    Retourne ces informations sous forme de chaîne de caractères.
    """
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        sys.exit(f"Erreur lors de la lecture du fichier CSV: {e}")
    
    metadata = f"Chemin du fichier CSV : {csv_file}\n"
    metadata += f"Nombre de lignes : {len(df)}\n"
    metadata += f"Nombre de colonnes : {len(df.columns)}\n\n"
    metadata += "Noms des colonnes :\n" + "\n".join(df.columns) + "\n\n"
    metadata += "Types des colonnes :\n" + "\n".join([f"{col} : {str(dtype)}" for col, dtype in df.dtypes.items()]) + "\n"
    
    return metadata

def extract_code(llm_output: str) -> str:
    """
    Extrait uniquement le code Python des blocs délimités par triple backticks.
    Si aucun bloc n'est trouvé, renvoie l'intégralité du texte.
    """
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", llm_output, re.DOTALL)
    if code_blocks:
        return "\n\n".join(code_blocks)
    else:
        return llm_output

def remove_shell_commands(code: str) -> str:
    """
    Filtre le code en supprimant les lignes ressemblant à des commandes shell (pip install, bash, etc.).
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

def attempt_execution_loop(code: str, metadata: str) -> None:
    """
    Tente d'exécuter le code. En cas d'erreur, envoie à Ollama le code fautif, les métadonnées
    et le message d'erreur, puis récupère le code corrigé et recommence jusqu'à obtenir une exécution sans erreur.
    """
    attempt = 0
    temp_filename = "temp_generated_script.py"
    while True:
        attempt += 1
        print(f"\n--- Tentative {attempt} ---")
        with open(temp_filename, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            result = subprocess.run(
                ["python3", temp_filename],
                capture_output=True,
                text=True,
                check=True
            )
            print("Sortie du script exécuté:")
            print(result.stdout)
            break  # Exécution réussie, sortie de la boucle
        except subprocess.CalledProcessError as e:
            print("Erreur lors de l'exécution du script généré:")
            print(e.stderr)
            # Création d'un nouveau prompt incluant les métadonnées, le message d'erreur et le code fautif
            recall_prompt = f"""Les informations suivantes concernent le fichier CSV :
{metadata}

Le code suivant a échoué lors de son exécution avec l'erreur suivante :
{e.stderr}

Merci de corriger le code suivant pour qu'il compile et s'exécute correctement :
--------------------------------------------------
{code}
--------------------------------------------------
"""
            print("Recall prompt envoyé à Ollama pour correction:")
            print(recall_prompt)
            try:
                recall_result = subprocess.run(
                    ["ollama", "run", "gemma3:27b", recall_prompt],
                    capture_output=True,
                    text=True,
                    check=True
                )
            except subprocess.CalledProcessError as e2:
                sys.exit(f"Erreur lors de l'appel à Ollama pour correction : {e2}")
            new_generated_script = recall_result.stdout
            # Extraction et nettoyage du nouveau code
            extracted_code = extract_code(new_generated_script)
            code = remove_shell_commands(extracted_code)
            print("\n--- Nouveau code extrait ---")
            print(code)
    print("Exécution réussie !")

def main():
    parser = argparse.ArgumentParser(
        description="Script pour générer des métadonnées d'un fichier CSV, appeler un LLM (Ollama) pour générer un script d'analyse économétrique et économique, nettoyer le code et l'exécuter en boucle jusqu'à correction complète."
    )
    parser.add_argument("csv_file", help="Chemin vers le fichier CSV")
    parser.add_argument("user_prompt", help="Prompt personnalisé pour l'appel LLM")
    parser.add_argument("--output", default="script_analyse.csv.py", help="Nom du fichier de script généré (par défaut: script_analyse.csv.py)")
    args = parser.parse_args()
    
    # Génération et enregistrement des métadonnées
    metadata = generate_metadata(args.csv_file)
    metadata_file = "metadata.txt"
    try:
        with open(metadata_file, "w", encoding="utf-8") as f:
            f.write(metadata)
        print(f"Métadonnées enregistrées dans '{metadata_file}'.")
    except Exception as e:
        sys.exit(f"Erreur lors de l'écriture des métadonnées : {e}")
    
    # Construction du prompt initial pour Ollama
    # On expose un éventail de méthodes d'analyse économétrique et économique que le LLM peut mobiliser,
    # sans le contraindre : 
    # - Régression multiple, régression logistique, séries temporelles, modèles de panel.
    # - Tests statistiques (normalité, multicolinéarité, etc.), diagnostics de régression.
    # - Analyses de tendance, cycles économiques, prévisions.
    # Le choix des méthodes doit être adapté aux caractéristiques et à la pertinence des données.
    prompt = f"""Les informations suivantes concernent un fichier CSV :
{metadata}

{args.user_prompt}

Génère un script Python complet qui réalise une analyse économétrique et économique approfondie à partir des données.
Le script doit être autonome et bien commenté. Tu disposes d'un large éventail de méthodes analytiques parmi lesquelles :
- Régressions (multiple, logistique, etc.), analyses de séries temporelles, modèles de panel.
- Tests statistiques (normalité, multicolinéarité, hétéroscédasticité, etc.) et diagnostics de régression.
- Analyses de tendances, cycles économiques et prévisions.
- Visualisations adaptées (histogrammes, boxplots, heatmaps, graphiques de résidus, etc.).

Je te laisse décider, en fonction de la nature et de la pertinence des données, quelles méthodes mobiliser et comment structurer l'analyse.
N'hésite pas à proposer des analyses supplémentaires si elles te semblent pertinentes.
N'oublie pas que le script doit utiliser le fichier CSV complet sans appliquer de transformations sur les données d'origine.
"""
    print("Prompt envoyé à Ollama :")
    print(prompt)
    
    # Appel initial à Ollama pour générer le script
    try:
        result = subprocess.run(
            ["ollama", "run", "gemma3:27b", prompt],
            capture_output=True,
            text=True,
            check=True
        )
        generated_script = result.stdout
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(generated_script)
        print(f"\nLe script généré a été enregistré dans '{args.output}'.")
    except subprocess.CalledProcessError as e:
        sys.exit(f"Erreur lors de l'appel à Ollama : {e}")
    
    # Extraction et nettoyage du code généré
    extracted_code = extract_code(generated_script)
    cleaned_code = remove_shell_commands(extracted_code)
    print("\n--- Code nettoyé extrait ---")
    print(cleaned_code)
    
    # Boucle d'exécution et de correction jusqu'à obtenir un code qui compile et s'exécute
    attempt_execution_loop(cleaned_code, metadata)

if __name__ == "__main__":
    main()
