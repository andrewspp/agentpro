#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import json
import logging
from datetime import datetime
import shutil

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pipeline")

def setup_directories():
    """Crée les répertoires nécessaires et nettoie ou fait pivoter les fichiers de log"""
    # S'assurer que le répertoire outputs existe pour les logs et les sorties JSON/PDF
    os.makedirs("outputs", exist_ok=True)
    
    # Obtenir le chemin absolu du répertoire courant
    current_dir = os.path.abspath(os.getcwd())
    
    # Nettoyer ou faire pivoter les fichiers de log existants
    for log_file in ["agent1.log", "agent2.log", "agent3.log"]:
        log_path = os.path.join(current_dir, log_file)
        try:
            # Créer des fichiers vides
            with open(log_path, 'w') as f:
                pass
            logger.info(f"Fichier de log {log_path} vidé")
            
            # Vérifier les permissions
            os.chmod(log_path, 0o666)  # Tous les droits de lecture/écriture
        except Exception as e:
            logger.warning(f"Problème avec le fichier de log {log_path}: {e}")
    
    logger.info("Répertoire 'outputs' créé et logs préparés")

def run_agent(agent_script, input_args, agent_name):
    """
    Exécute un agent et retourne son fichier de sortie JSON.
    """
    logger.info(f"Démarrage de l'agent: {agent_name}")
    
    # Vérifier si --model et --backend sont dans les arguments du pipeline
    if "--model" not in input_args:
        try:
            model_index = sys.argv.index("--model")
            model_value = sys.argv[model_index + 1]
            input_args.extend(["--model", model_value])
        except (ValueError, IndexError):
            logger.warning("Modèle non spécifié via --model, utilisation du défaut de l'agent.")
    
    # Ajout du backend (gemini ou ollama)
    if "--backend" not in input_args:
        try:
            backend_index = sys.argv.index("--backend")
            backend_value = sys.argv[backend_index + 1]
            input_args.extend(["--backend", backend_value])
        except (ValueError, IndexError):
            logger.warning("Backend non spécifié via --backend, utilisation du défaut de l'agent.")

    # Ajouter l'argument de log forcé pour chaque agent
    input_args.extend(["--log-file", f"{agent_name}.log"])
    
    logger.info(f"Commande pour {agent_name}: python3 {agent_script} {' '.join(input_args)}")

    try:
        # Obtenir le répertoire courant pour s'assurer que les chemins sont corrects
        current_dir = os.path.abspath(os.getcwd())
        
        # Créer un environnement qui force le logging immédiat
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        # Exécuter l'agent avec stdout capturé
        process = subprocess.Popen(
            ["python3", agent_script] + input_args,
            stdout=subprocess.PIPE,
            stderr=None,  # Pas de capture de stderr - laisse le écrire directement
            text=True,
            env=env,
            cwd=current_dir
        )
        
        stdout, _ = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"L'agent {agent_name} a échoué avec le code de retour {process.returncode}")
            sys.exit(f"Échec de l'agent {agent_name}")
        
        # Vérifier si la sortie est du JSON valide avant d'écrire
        try:
            json.loads(stdout)
        except json.JSONDecodeError as json_err:
            logger.error(f"La sortie de {agent_name} n'est pas du JSON valide: {json_err}")
            logger.error(f"Sortie reçue (premiers 500 chars):\n{stdout[:500]}")
            # Écrire quand même pour le débogage, mais avec une extension différente
            output_file = os.path.join("outputs", f"{agent_name}_output_invalid.txt")
            with open(output_file, "w", encoding="utf-8") as f:
                 f.write(stdout)
            logger.error(f"Sortie brute enregistrée dans {output_file}")
            raise ValueError(f"Sortie invalide de {agent_name}") # Provoquer une erreur

        output_file = os.path.join("outputs", f"{agent_name}_output.json")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(stdout)

        logger.info(f"Agent {agent_name} terminé avec succès. Sortie enregistrée dans {output_file}")
        return output_file # Retourner le chemin du fichier JSON
    except Exception as ex:
         logger.error(f"Erreur lors de l'exécution de {agent_name}: {ex}")
         sys.exit(f"Erreur pour {agent_name}: {ex}")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline académique multi-agent pour l'analyse économétrique"
    )
    parser.add_argument("csv_file", help="Chemin vers le fichier CSV")
    parser.add_argument("user_prompt", help="Prompt utilisateur décrivant l'analyse souhaitée")
    parser.add_argument("--model", default="gemma3:27b", help="Modèle LLM à utiliser (défaut: gemma3:27b)")
    parser.add_argument("--backend", default="ollama", choices=["ollama", "gemini"], 
                        help="Backend LLM à utiliser: 'ollama' (défaut) ou 'gemini'")
    parser.add_argument("--academic", action="store_true", default=True, 
                        help="Générer un rapport au format académique (activé par défaut)")
    args = parser.parse_args()

    # Conversion du chemin relatif en chemin absolu
    csv_file_absolute = os.path.abspath(args.csv_file)
    if not os.path.exists(csv_file_absolute):
        logger.error(f"Le fichier CSV spécifié n'existe pas : {csv_file_absolute}")
        sys.exit(1)
    logger.info(f"Chemin absolu du fichier CSV: {csv_file_absolute}")

    # Création des répertoires et préparation des logs
    setup_directories()

    # Exécution de l'agent 1 (Ingestion et problématisation académique)
    logger.info("=== DÉMARRAGE DE L'AGENT 1: INGESTION ET CONCEPTUALISATION ACADÉMIQUE ===")
    agent1_output_file = run_agent(
        "agent1.py",
        [csv_file_absolute, args.user_prompt, "--model", args.model, "--backend", args.backend],
        "agent1"
    )

    # Exécution de l'agent 2 (Analyse économétrique académique)
    logger.info("=== DÉMARRAGE DE L'AGENT 2: ANALYSE ÉCONOMÉTRIQUE ACADÉMIQUE ===")
    agent2_output_file = run_agent(
        "agent2.py",
        [csv_file_absolute, args.user_prompt, agent1_output_file, "--model", args.model, "--backend", args.backend],
        "agent2"
    )

    # Exécution de l'agent 3 (Synthèse académique et rapport)
    logger.info("=== DÉMARRAGE DE L'AGENT 3: SYNTHÈSE ACADÉMIQUE ET RAPPORT ===")
    agent3_output_file = run_agent(
        "agent3.py",
        # Passer les sorties des agents 1 et 2, et le prompt original
        [agent1_output_file, agent2_output_file, args.user_prompt, "--model", args.model, "--backend", args.backend],
        "agent3"
    )

    # Affichage du résultat final
    try:
        with open(agent3_output_file, "r", encoding="utf-8") as f:
            result = json.load(f)

        logger.info("=== PIPELINE ACADÉMIQUE TERMINÉE AVEC SUCCÈS ===")
        print("\n" + "="*80)
        print("RÉSUMÉ ACADÉMIQUE:")
        print("="*80)
        print(result.get("abstract", "Aucun résumé académique généré"))
        print("\n" + "="*80)
        
        if result.get("rapport_pdf"):
            print(f"Rapport académique complet généré: {result.get('rapport_pdf')}")
            # Ouvrir automatiquement le PDF si possible
            try:
                import platform
                
                system = platform.system()
                pdf_path = result.get('rapport_pdf')
                
                if system == 'Windows':
                    os.startfile(pdf_path)
                elif system == 'Darwin':  # macOS
                    subprocess.call(['open', pdf_path])
                elif system == 'Linux':
                    subprocess.call(['xdg-open', pdf_path])
                
                print("Le rapport PDF a été ouvert automatiquement.")
            except Exception as e:
                print(f"Le rapport a été généré mais n'a pas pu être ouvert automatiquement: {e}")
        else:
            print("ERREUR : Le rapport PDF n'a pas pu être généré.")
            print(f"Détails de l'erreur : {result.get('error', 'Inconnus')}")
            
        print("="*80)

    except FileNotFoundError:
         logger.error(f"Le fichier de sortie de l'agent 3 ({agent3_output_file}) est introuvable.")
         sys.exit("Échec final du pipeline académique.")
    except json.JSONDecodeError:
         logger.error(f"Le fichier de sortie de l'agent 3 ({agent3_output_file}) contient du JSON invalide.")
         sys.exit("Échec final du pipeline académique.")
    except Exception as e:
         logger.error(f"Erreur lors de la lecture du résultat final: {e}")
         sys.exit("Échec final du pipeline académique.")


if __name__ == "__main__":
    main()