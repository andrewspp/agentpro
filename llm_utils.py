#!/usr/bin/env python3
"""
Utilitaires pour l'appel aux modèles de langage (LLM).
Supporte à la fois les backends Ollama (local) et Gemini (API cloud).

Améliorations:
- Robustesse accrue de l'appel API REST Gemini pour les images.
- Logging détaillé des requêtes/réponses Gemini.
- Option de débogage via variable d'environnement pour forcer l'API REST.
- Gestion des réponses bloquées par Gemini.
- Timeouts configurables.
"""

import subprocess
import os
import requests
import base64
import json
import logging
import time

# Configuration du logging
# Use specific format including module name and line number for better debugging
log_format = '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("llm_utils")

# --- Configuration ---
# ⛔️ Clé API intégrée directement ici (⚠️ usage local uniquement, ne jamais versionner en prod)
# Il est FORTEMENT recommandé de charger la clé depuis une variable d'environnement
# ou un système de gestion de secrets en production.
_DEFAULT_GEMINI_API_KEY = "AIzaSyAYT-NjrJiRK9Ei8gp716uR57CO59puWhg" # Replace with your actual key if needed

# Injecte la clé dans l'environnement si elle n'est pas déjà définie
if "GEMINI_API_KEY" not in os.environ:
    os.environ["GEMINI_API_KEY"] = _DEFAULT_GEMINI_API_KEY
    logger.warning("Utilisation de la clé API Gemini par défaut intégrée au code. À utiliser pour le développement local UNIQUEMENT.")
if "GOOGLE_API_KEY" not in os.environ:
     os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"] # Pour compatibilité avec la bibliothèque google.generativeai

# --- Constantes ---
DEFAULT_OLLAMA_MODEL = "qwq:32b"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash-latest" # Utilisation du modèle Flash plus récent
DEFAULT_GEMINI_TIMEOUT = 90 # Secondes (augmenté pour analyse d'image)

# --- Variables de contrôle (pour le débogage) ---
# Mettre à True via variable d'environnement pour forcer l'API REST pour les images
FORCE_GEMINI_REST_FOR_IMAGES = True
# FORCE_GEMINI_REST_FOR_IMAGES = os.getenv('LLM_UTILS_FORCE_GEMINI_REST', 'false').lower() == 'true' # Ligne originale commentée
if FORCE_GEMINI_REST_FOR_IMAGES:
     logger.warning("LLM_UTILS_FORCE_GEMINI_REST=true: Forçage de l'utilisation de l'API REST pour les requêtes Gemini avec image.")

# --- Fonctions principales ---

def call_llm(prompt, model_name=None, backend="ollama", image_base64=None, timeout=None):
    """
    Appelle un modèle LLM selon le backend spécifié (ollama ou gemini).
    Prend en charge l'envoi d'images pour Gemini.

    Args:
        prompt (str): Texte du prompt à envoyer.
        model_name (str, optional): Nom du modèle à utiliser. Défaut selon le backend.
        backend (str): "ollama" ou "gemini". Défaut: "ollama".
        image_base64 (str, optional): Données image encodées en base64 (uniquement pour gemini).
        timeout (int, optional): Timeout en secondes pour les appels API Gemini. Défaut: DEFAULT_GEMINI_TIMEOUT.

    Returns:
        str: Réponse du modèle.

    Raises:
        ValueError: Si le backend n'est pas reconnu ou si la clé API est manquante pour Gemini.
        RuntimeError: Si l'appel au backend échoue (Ollama ou Gemini).
        ImportError: Si la bibliothèque google.generativeai est nécessaire mais non installée.
    """
    logger.debug(f"Appel call_llm - backend: {backend}, modèle: {model_name or 'défaut'}, image: {'Oui' if image_base64 else 'Non'}")

    if backend == "ollama":
        if image_base64:
            logger.warning("Le backend Ollama ne prend pas en charge les images. L'image sera ignorée.")
        model = model_name or DEFAULT_OLLAMA_MODEL
        return call_ollama(model, prompt)

    elif backend == "gemini":
        model = model_name or DEFAULT_GEMINI_MODEL
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("[llm_utils] Clé API Gemini (GEMINI_API_KEY) manquante dans l'environnement.")

        effective_timeout = timeout if timeout is not None else DEFAULT_GEMINI_TIMEOUT
        return call_gemini(model, prompt, api_key, image_base64, effective_timeout)

    else:
        raise ValueError(f"[llm_utils] Backend non reconnu: '{backend}' (attendu: 'ollama' ou 'gemini')")

def call_ollama(model, prompt):
    """
    Appelle un modèle local via la commande Ollama CLI.

    Args:
        model (str): Nom du modèle Ollama.
        prompt (str): Texte du prompt.

    Returns:
        str: Réponse du modèle (stdout).

    Raises:
        RuntimeError: Si la commande ollama échoue.
    """
    command = ["ollama", "run", model, prompt]
    logger.info(f"Exécution Ollama: {' '.join(command)}")
    try:
        start_time = time.time()
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8' # Assurer l'encodage correct
        )
        duration = time.time() - start_time
        logger.info(f"Appel Ollama réussi ({duration:.2f}s).")
        logger.debug(f"Ollama stdout:\n{result.stdout[:500]}{'...' if len(result.stdout)>500 else ''}")
        return result.stdout

    except FileNotFoundError:
        logger.error("Commande 'ollama' non trouvée. Assurez-vous qu'Ollama est installé et dans le PATH.")
        raise RuntimeError("[llm_utils] Commande 'ollama' non trouvée.")
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        logger.error(f"Erreur lors de l'appel Ollama ({duration:.2f}s). stderr:\n{e.stderr.strip()}")
        raise RuntimeError(f"[llm_utils] Erreur Ollama (code {e.returncode}): {e.stderr.strip()}")
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Erreur inattendue lors de l'appel Ollama ({duration:.2f}s): {e}", exc_info=True)
        raise RuntimeError(f"[llm_utils] Erreur inattendue Ollama: {e}")


def call_gemini(model, prompt, api_key, image_base64=None, timeout=DEFAULT_GEMINI_TIMEOUT):
    """
    Appelle l'API Gemini de Google, gérant les images via la bibliothèque ou l'API REST.

    Args:
        model (str): Nom du modèle Gemini (ex: "gemini-1.5-flash-latest").
        prompt (str): Texte du prompt.
        api_key (str): Clé API Google.
        image_base64 (str, optional): Image encodée en base64.
        timeout (int): Timeout en secondes pour l'appel API.

    Returns:
        str: Réponse textuelle du modèle.

    Raises:
        RuntimeError: Si l'appel API échoue ou la réponse est invalide.
        ImportError: Si la bibliothèque google.generativeai est requise mais non trouvée.
    """
    if image_base64:
        # Décider quelle méthode utiliser pour l'appel avec image
        use_rest_api = FORCE_GEMINI_REST_FOR_IMAGES
        if not use_rest_api:
            try:
                # Tenter d'utiliser la bibliothèque officielle
                logger.info(f"Tentative d'appel Gemini avec image via la bibliothèque google.generativeai (modèle: {model})")
                return call_gemini_with_image_via_library(model, prompt, image_base64, api_key, timeout)
            except ImportError:
                logger.warning("Bibliothèque google.generativeai non disponible ou import échoué. Utilisation de l'API REST pour l'image.")
                use_rest_api = True
            except Exception as lib_err:
                 # Si la bibliothèque est installée mais échoue pour une autre raison,
                 # on peut choisir de passer à REST ou de lever l'erreur.
                 # Ici, on passe à REST pour maximiser les chances de succès.
                logger.error(f"Erreur inattendue avec la bibliothèque google.generativeai: {lib_err}. Tentative avec l'API REST.", exc_info=True)
                use_rest_api = True

        if use_rest_api:
            # Utiliser l'API REST comme fallback ou si forcé
             logger.info(f"Appel Gemini avec image via l'API REST (modèle: {model})")
             return call_gemini_with_image_via_rest(model, prompt, image_base64, api_key, timeout)

    else:
        # Appel API REST standard sans image
        logger.info(f"Appel Gemini standard (texte seul) via l'API REST (modèle: {model})")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {"contents": [{"parts": [{"text": prompt}]}]} # Structure simple pour texte seul

        try:
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            duration = time.time() - start_time
            logger.info(f"Appel API Gemini (texte seul) terminé - Statut: {response.status_code} ({duration:.2f}s)")

            response.raise_for_status() # Lève une exception pour les erreurs HTTP 4xx/5xx
            data = response.json()
            logger.debug(f"Réponse JSON Gemini (texte seul):\n{json.dumps(data, indent=2)}")

            # Vérifier les erreurs applicatives dans la réponse JSON
            if 'error' in data:
                logger.error(f"Erreur API Gemini dans la réponse JSON: {json.dumps(data['error'], indent=2)}")
                raise RuntimeError(f"[llm_utils] Erreur API Gemini: {data['error'].get('message', 'Erreur inconnue')}")

            # Extraction robuste du texte
            text = extract_text_from_gemini_response(data)
            return text

        except requests.exceptions.Timeout:
            duration = time.time() - start_time
            logger.error(f"Timeout dépassé ({timeout}s) pour l'appel API Gemini (texte seul).")
            raise RuntimeError(f"[llm_utils] Timeout API Gemini ({timeout}s) dépassé.")
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"Erreur de requête API Gemini (texte seul) ({duration:.2f}s): {e}", exc_info=True)
            raise RuntimeError(f"[llm_utils] Erreur de requête API Gemini: {e}")
        except Exception as e:
            duration = time.time() - start_time if 'start_time' in locals() else 0
            logger.error(f"Erreur inattendue lors de l'appel Gemini (texte seul) ({duration:.2f}s): {e}", exc_info=True)
            raise RuntimeError(f"[llm_utils] Erreur inattendue Gemini: {e}")


def call_gemini_with_image_via_library(model, prompt, image_base64, api_key, timeout):
    """
    Utilise la bibliothèque officielle google.generativeai pour envoyer une requête avec image.
    NOTE: Cette fonction lève ImportError si la bibliothèque n'est pas trouvée.
          Elle peut aussi lever d'autres exceptions de la bibliothèque elle-même.
    """
    try:
        import google.generativeai as genai
        # from PIL import Image # PIL peut être nécessaire pour certains formats/validations
        import io
    except ImportError as e:
        logger.error(f"Dépendance manquante: {e}. Installez google-generativeai.")
        raise ImportError(f"Module google.generativeai non disponible. Installez-le (pip install google-generativeai): {e}") from e

    genai.configure(api_key=api_key)

    try:
        # Décoder l'image base64 en bytes
        image_bytes = base64.b64decode(image_base64)
        logger.debug(f"Image base64 décodée en {len(image_bytes)} octets.")

        # Préparer l'objet image pour la bibliothèque
        # Utiliser 'blob' semble être la méthode recommandée pour les bytes
        img_blob = {'mime_type': 'image/png', 'data': image_bytes}

        # Créer le contenu multimodal
        contents = [prompt, img_blob]
        logger.debug(f"Contenu envoyé à la bibliothèque Gemini: [prompt, {img_blob['mime_type']} ({len(img_blob['data'])} bytes)]")

        # Créer le modèle Gemini
        # Ajout de configuration de génération si nécessaire (température, etc.)
        generation_config = genai.types.GenerationConfig(
            # candidate_count=1, # Généralement 1 par défaut
            # stop_sequences=['...'],
            # max_output_tokens=2048,
            temperature=0.5, # Ajuster si besoin
            # top_p=1.0,
            # top_k=1
        )

        # Note: La bibliothèque ne semble pas avoir de timeout direct sur generate_content
        # Le timeout global de la requête HTTP sous-jacente pourrait s'appliquer.
        model_gemini = genai.GenerativeModel(model)

        logger.info("Appel à model.generate_content() via la bibliothèque...")
        start_time = time.time()
        response = model_gemini.generate_content(contents, generation_config=generation_config)
        duration = time.time() - start_time
        logger.info(f"Appel bibliothèque Gemini terminé ({duration:.2f}s).")

        # Log de la réponse brute pour débogage
        try:
            # Tenter d'accéder à des attributs courants pour le log
            response_summary = f"Response(text_len={len(response.text) if hasattr(response, 'text') else 'N/A'}, finish_reason={getattr(response, 'prompt_feedback', {}).get('block_reason', 'N/A')})"
            logger.debug(f"Réponse brute de la bibliothèque Gemini: {response_summary}")
            # logger.debug(f"Full response object: {response}") # Attention, peut être très verbeux
        except Exception as log_err:
            logger.warning(f"Impossible de générer le résumé de la réponse brute: {log_err}")

        # Vérifier si la réponse a été bloquée
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            safety_ratings = getattr(response, 'candidates', [{}])[0].get('safety_ratings', 'N/A')
            logger.error(f"Réponse Gemini bloquée par la bibliothèque. Raison: {block_reason}, Safety Ratings: {safety_ratings}")
            raise RuntimeError(f"[llm_utils] Réponse Gemini bloquée (lib): {block_reason}")

        # Extraire et retourner le texte
        if hasattr(response, 'text') and response.text:
            logger.info("Texte extrait avec succès de la réponse de la bibliothèque.")
            return response.text
        else:
            # Gérer le cas où .text est absent ou vide, peut arriver si bloqué ou autre erreur
            logger.error(f"Aucun attribut 'text' trouvé ou vide dans la réponse de la bibliothèque. Réponse: {response}")
            raise RuntimeError("[llm_utils] Format de réponse inattendu de la bibliothèque Gemini (pas de texte).")

    except ImportError: # Re-lever pour que l'appelant sache qu'il faut utiliser REST
         raise
    except Exception as e:
        # Capturer d'autres erreurs potentielles de la bibliothèque (ex: APIError, InvalidArgument)
        logger.error(f"Erreur lors de l'appel à la bibliothèque Gemini: {type(e).__name__}: {e}", exc_info=True)
        # Renvoyer l'erreur pour éventuellement tenter avec REST API
        raise RuntimeError(f"[llm_utils] Erreur avec google.generativeai: {e}") from e


def call_gemini_with_image_via_rest(model, prompt, image_base64, api_key, timeout):
    """
    Utilise l'API REST de Gemini pour envoyer une requête avec image.
    Inclut un logging et une gestion d'erreurs robustes.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}

    # Création du payload avec texte et image
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png", "data": image_base64}}
            ]
        }],
        # Ajouter generationConfig si nécessaire (température, max tokens etc.)
        "generationConfig": {
             "temperature": 0.5,
             "maxOutputTokens": 4096 # Augmenter pour les descriptions d'images potentiellement longues
        #     "topP": 0.8,
        #     "topK": 10
        }
    }
    logger.debug(f"Payload JSON envoyé à l'API REST Gemini (image):\n{json.dumps(payload, indent=2, default=lambda x: f'<bytes {len(x)}>' if isinstance(x, bytes) else f'<base64 {len(x)}>')}") # Ne pas logger l'image entière

    try:
        start_time = time.time()
        response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        duration = time.time() - start_time

        # Log systématique de la réponse pour débogage
        logger.info(f"Appel API REST Gemini (image) terminé - Statut: {response.status_code} ({duration:.2f}s)")
        logger.debug(f"Réponse Headers: {response.headers}")
        try:
            # Tenter de décoder en JSON pour le log, même si erreur HTTP
            data = response.json()
            logger.debug(f"Réponse JSON Body:\n{json.dumps(data, indent=2)}")
        except json.JSONDecodeError:
            logger.error(f"La réponse API n'est pas au format JSON. Contenu brut:\n{response.text[:1000]}{'...' if len(response.text)>1000 else ''}")
            # Si ce n'est pas JSON, et que le statut n'est pas 2xx, raise_for_status lèvera l'erreur.
            # Si le statut est 2xx mais que ce n'est pas JSON, c'est une erreur inattendue.
            if 200 <= response.status_code < 300:
                raise RuntimeError("[llm_utils] Réponse API Gemini (REST) inattendue: Statut 2xx mais pas de JSON.")

        # Vérifier les erreurs HTTP (4xx, 5xx) après avoir loggé la réponse
        response.raise_for_status()

        # Si on arrive ici, status code est 2xx et la réponse est JSON
        # Vérifier les erreurs applicatives dans la réponse JSON
        if 'error' in data:
            logger.error(f"Erreur API Gemini détectée dans la réponse JSON: {json.dumps(data['error'], indent=2)}")
            raise RuntimeError(f"[llm_utils] Erreur API Gemini (REST): {data['error'].get('message', 'Erreur inconnue')}")

        # Extraction robuste du texte de la réponse
        text = extract_text_from_gemini_response(data)
        return text

    except requests.exceptions.Timeout:
        duration = time.time() - start_time
        logger.error(f"Timeout dépassé ({timeout}s) pour l'appel API REST Gemini (image).")
        raise RuntimeError(f"[llm_utils] Timeout API Gemini ({timeout}s) dépassé.")
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time if 'start_time' in locals() else 0
        logger.error(f"Erreur de requête API REST Gemini (image) ({duration:.2f}s): {e}", exc_info=True)
        raise RuntimeError(f"[llm_utils] Erreur de requête API Gemini (REST): {e}")
    except Exception as e: # Capture toute autre erreur (JSONDecodeError si status 2xx, RuntimeError interne, etc.)
        duration = time.time() - start_time if 'start_time' in locals() else 0
        logger.error(f"Erreur inattendue lors de l'appel API REST Gemini (image) ({duration:.2f}s): {e}", exc_info=True)
        # Si ce n'est pas déjà une RuntimeError, on l'enveloppe
        if isinstance(e, RuntimeError):
            raise e
        else:
            raise RuntimeError(f"[llm_utils] Erreur inattendue Gemini (REST): {e}")


def extract_text_from_gemini_response(data):
    """
    Extrait le texte de manière robuste d'une réponse JSON Gemini réussie.

    Args:
        data (dict): Le dictionnaire JSON de la réponse Gemini.

    Returns:
        str: Le texte extrait.

    Raises:
        RuntimeError: Si la structure de la réponse est invalide ou si le contenu est bloqué.
    """
    try:
        if not data or 'candidates' not in data or not isinstance(data['candidates'], list) or len(data['candidates']) == 0:
            logger.error(f"Structure de réponse Gemini invalide: 'candidates' manquante, non-liste ou vide. Réponse: {json.dumps(data, indent=2)}")
            raise RuntimeError("[llm_utils] Format de réponse Gemini invalide: 'candidates' incorrect.")

        candidate = data['candidates'][0]

        # Vérifier si la génération s'est terminée correctement ou a été bloquée
        finish_reason = candidate.get('finishReason')
        if finish_reason and finish_reason != 'STOP':
            safety_ratings = candidate.get('safetyRatings', [])
            logger.warning(f"Réponse Gemini potentiellement incomplète ou bloquée. FinishReason: {finish_reason}. SafetyRatings: {safety_ratings}")
            # Selon le cas d'usage, on peut vouloir lever une erreur ou retourner un message spécifique.
            # Ici, on lève une erreur pour signaler clairement le problème.
            # On pourrait aussi vérifier si 'content' existe malgré le finishReason.
            raise RuntimeError(f"[llm_utils] Réponse Gemini bloquée ou incomplète ({finish_reason}).")
            # Alternative : return f"[Contenu bloqué ou incomplet: {finish_reason}]"

        if 'content' not in candidate or 'parts' not in candidate['content'] or not isinstance(candidate['content']['parts'], list):
            logger.error(f"Structure de réponse Gemini invalide: 'content' ou 'parts' manquants ou incorrects. Candidat: {json.dumps(candidate, indent=2)}")
            raise RuntimeError("[llm_utils] Format de réponse Gemini invalide: 'content' ou 'parts' incorrects.")

        if len(candidate['content']['parts']) == 0:
            logger.warning(f"La section 'parts' de la réponse Gemini est vide. Candidat: {json.dumps(candidate, indent=2)}")
            return "" # Retourner une chaîne vide si parts est vide mais la réponse est valide

        # Extrait le texte de la première partie (le format habituel)
        first_part = candidate['content']['parts'][0]
        if 'text' not in first_part:
            logger.warning(f"Aucune clé 'text' trouvée dans la première partie de la réponse Gemini. Parts: {json.dumps(candidate['content']['parts'], indent=2)}")
            # Peut-être que la réponse est dans une autre partie ? Ou pas de texte du tout.
            # Pour l'instant, on retourne une chaîne vide.
            return ""

        text = first_part['text']
        logger.info(f"Texte extrait avec succès de la réponse Gemini (longueur: {len(text)}).")
        return text

    except (KeyError, IndexError, TypeError) as e:
        logger.error(f"Erreur lors de l'analyse de la structure de la réponse Gemini: {e}. Réponse complète: {json.dumps(data, indent=2)}", exc_info=True)
        raise RuntimeError(f"[llm_utils] Format de réponse Gemini inattendu lors de l'extraction: {e}")


# --- Test ---
if __name__ == "__main__":
    logger.info("--- Début des tests llm_utils ---")

    # === Test 1: Ollama simple ===
    print("\n--- Test 1: Appel Ollama simple ---")
    try:
        # Définir explicitement le modèle pour le test si nécessaire
        test_ollama_model = DEFAULT_OLLAMA_MODEL # Ou un modèle plus petit comme "phi3" ou "llama3:8b" si qwq:32b est trop gros/lent
        response_ollama = call_llm("Explique la loi d'Ohm en une phrase.", backend="ollama", model_name=test_ollama_model)
        print(f"Réponse Ollama ({test_ollama_model}):\n{response_ollama}")
    except Exception as e:
        print(f"Erreur lors du test Ollama: {e}")
        logger.warning("Le test Ollama a échoué. Vérifiez que Ollama est en cours d'exécution et que le modèle est disponible.")

    # === Test 2: Gemini simple (texte) ===
    print("\n--- Test 2: Appel Gemini simple (texte) ---")
    try:
        response_gemini_text = call_llm("Quelle est la capitale de l'Australie?", backend="gemini")
        print(f"Réponse Gemini (texte):\n{response_gemini_text}")
    except Exception as e:
        print(f"Erreur lors du test Gemini (texte): {e}")

    # === Test 3: Gemini avec image (API REST forcée si variable définie) ===
    print("\n--- Test 3: Appel Gemini avec image ---")
    # Créer une image simple en base64 pour le test (carré rouge 10x10 px)
    # Vous pouvez remplacer ceci par la lecture d'un vrai fichier image si besoin
    # Note: Une vraie image est préférable pour un test réaliste.
    dummy_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mP8z8AARMDUwMBWAQC8kgPb2fU7+QAAAABJRU5ErkJggg==" # 10x10 red PNG

    # Tentative de lecture d'une image locale si elle existe
    test_image_path = "test_image.png" # Mettez le chemin vers une image PNG de test ici
    try:
        if os.path.exists(test_image_path):
             with open(test_image_path, "rb") as f:
                 image_bytes = f.read()
                 image_base64_to_use = base64.b64encode(image_bytes).decode('utf-8')
                 logger.info(f"Utilisation de l'image de test locale: {test_image_path} ({len(image_bytes)} bytes)")
        else:
            logger.warning(f"Image de test locale '{test_image_path}' non trouvée. Utilisation de l'image dummy.")
            image_base64_to_use = dummy_image_base64
    except Exception as e:
        logger.error(f"Erreur lors de la lecture de l'image de test locale: {e}. Utilisation de l'image dummy.")
        image_base64_to_use = dummy_image_base64

    try:
        if not os.getenv("GEMINI_API_KEY"):
             print("Skipping test Gemini image: Clé API non configurée.")
        else:
            response_gemini_image = call_llm(
                "Décris brièvement cette image.",
                backend="gemini",
                image_base64=image_base64_to_use
            )
            print(f"Réponse Gemini (image):\n{response_gemini_image}")
    except Exception as e:
        print(f"Erreur lors du test Gemini (image): {e}")

    logger.info("--- Fin des tests llm_utils ---")