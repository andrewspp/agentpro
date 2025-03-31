#!/usr/bin/env python3
"""
Documentation Generator pour l'agent d'analyse économétrique

Ce script génère un document PDF complet qui détaille l'architecture,
le fonctionnement et le code source de l'agent d'analyse économétrique.
"""

import os
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted, Image
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

# Configuration
OUTPUT_PDF = "Agent_Econometrique_Documentation.pdf"
SOURCE_FILES = [
    "pipeline.py",
    "agent1.py",
    "agent2.py",
    "agent3.py",
    "llm_utils.py"
]

# Structure du document
DOCUMENT_STRUCTURE = [
    {
        "title": "Documentation Complète de l'Agent d'Analyse Économétrique",
        "type": "heading",
        "level": 0
    },
    {
        "title": "1. Architecture Globale",
        "type": "heading",
        "level": 1,
        "content": """
L'agent d'analyse économétrique est un système multi-agent composé de trois agents spécialisés qui travaillent ensemble pour réaliser une analyse économétrique complète à partir d'un fichier CSV et d'une requête utilisateur. L'architecture est conçue pour être modulaire, permettant à chaque agent de se concentrer sur une tâche spécifique dans le pipeline d'analyse.

Le système est orchestré par un script principal (pipeline.py) qui coordonne l'exécution séquentielle des agents. Chaque agent produit un résultat qui sert d'entrée à l'agent suivant, formant ainsi une chaîne de traitement cohérente.

Voici les composants principaux du système:

1. Pipeline (pipeline.py) - Orchestrateur qui gère l'exécution des agents
2. Agent 1 (agent1.py) - Ingestion des données et problématisation académique
3. Agent 2 (agent2.py) - Analyse économétrique et génération de visualisations
4. Agent 3 (agent3.py) - Synthèse et génération du rapport final
5. Utilitaires LLM (llm_utils.py) - Fonctions pour interagir avec les modèles de langage

Le système utilise des modèles de langage (LLMs) à travers deux backends possibles:
- Ollama: pour l'exécution locale des modèles
- Gemini: pour l'accès aux API de Google

Cette architecture permet de traiter des données économiques, de générer des analyses statistiques rigoureuses, et de produire des rapports académiques complets avec visualisations et interprétations."""
    },
    {
        "title": "2. Workflow et Flux de Données",
        "type": "heading",
        "level": 1,
        "content": """
Le workflow de l'agent d'analyse économétrique suit un flux séquentiel où chaque étape dépend des résultats de l'étape précédente:

1. L'utilisateur fournit un fichier CSV contenant des données économiques et une requête décrivant l'analyse souhaitée.

2. Le pipeline initialise l'environnement et orchestre l'exécution des agents.

3. L'Agent 1 (Ingestion et Problématisation):
   - Charge le fichier CSV et extrait des métadonnées détaillées
   - Détecte les problèmes potentiels dans les données
   - Conceptualise la question de recherche dans un cadre académique
   - Génère une introduction, une revue de littérature, des hypothèses et une méthodologie
   - Produit un fichier JSON contenant ces éléments

4. L'Agent 2 (Analyse Économétrique):
   - Utilise les résultats de l'Agent 1 comme contexte
   - Génère du code Python pour l'analyse statistique
   - Exécute ce code, en corrigeant automatiquement les erreurs si nécessaire
   - Produit des visualisations, des tableaux de régression et des analyses statistiques
   - Interprète les résultats avec l'aide du LLM
   - Génère un fichier JSON contenant les résultats et les interprétations

5. L'Agent 3 (Synthèse et Rapport):
   - Combine les résultats des Agents 1 et 2
   - Génère une synthèse globale des résultats
   - Produit un document PDF bien structuré et formaté
   - Inclut toutes les visualisations avec leurs interprétations
   - Génère également un document Word si les bibliothèques requises sont disponibles

6. Le pipeline présente les résultats finaux à l'utilisateur et ouvre automatiquement le rapport PDF.

Flux de données:
CSV + Requête Utilisateur → Agent 1 → JSON → Agent 2 → JSON → Agent 3 → PDF/DOCX

Chaque agent communique avec les modèles de langage via le module llm_utils.py, qui fournit une interface unifiée pour interagir avec différents backends (Ollama pour l'exécution locale, Gemini pour l'API cloud)."""
    },
    {
        "title": "3. Détail des Composants",
        "type": "heading",
        "level": 1
    },
    {
        "title": "3.1 Pipeline Principal (pipeline.py)",
        "type": "heading",
        "level": 2,
        "content": """
Le pipeline.py est le script principal qui orchestre l'exécution de l'ensemble du système. Il prend en charge:

- Le parsing des arguments de ligne de commande
- La préparation de l'environnement (création des répertoires, initialisation des logs)
- L'exécution séquentielle des trois agents
- La gestion des erreurs et la communication entre les agents
- L'affichage des résultats finaux et l'ouverture automatique du rapport

Le pipeline garantit que chaque agent reçoit les entrées nécessaires et que les sorties sont correctement formatées pour l'agent suivant. Il maintient également des logs détaillés pour faciliter le débogage."""
    },
    {
        "title": "3.2 Agent 1: Ingestion et Problématisation (agent1.py)",
        "type": "heading",
        "level": 2,
        "content": """
L'Agent 1 est responsable de l'ingestion des données et de la problématisation académique. Ses fonctions principales sont:

- Lecture et analyse du fichier CSV pour extraire des métadonnées détaillées
- Détection des problèmes potentiels dans les données (valeurs manquantes, anomalies)
- Utilisation d'un LLM pour conceptualiser la question de recherche dans un cadre académique rigoureux
- Génération d'une introduction, d'une revue de littérature, d'hypothèses formelles et d'une méthodologie
- Analyse des limites méthodologiques et des variables clés
- Production d'un fichier JSON structuré contenant toutes ces informations

L'Agent 1 pose les fondements conceptuels de l'analyse, en transformant une requête utilisateur simple en un cadre de recherche académique rigoureux."""
    },
    {
        "title": "3.3 Agent 2: Analyse Économétrique (agent2.py)",
        "type": "heading",
        "level": 2,
        "content": """
L'Agent 2 est le cœur analytique du système, responsable de la génération et de l'exécution du code d'analyse économétrique. Ses fonctions principales sont:

- Génération de code Python pour l'analyse statistique et économétrique
- Exécution du code avec gestion automatique des erreurs et tentatives de correction
- Production de visualisations (graphiques, nuages de points, matrices de corrélation)
- Génération de modèles de régression et interprétation des résultats
- Capture et interprétation des sorties de visualisation
- Gestion robuste des erreurs avec tentatives multiples de correction, y compris l'utilisation de modèles plus puissants
- Production d'un fichier JSON contenant tous les résultats, visualisations et interprétations

L'Agent 2 est particulièrement sophistiqué, car il doit non seulement générer du code, mais aussi l'exécuter, capturer les erreurs, les corriger, et interpréter les résultats. Il inclut également des mécanismes de fallback vers des modèles plus puissants en cas d'échec répété."""
    },
    {
        "title": "3.4 Agent 3: Synthèse et Rapport (agent3.py)",
        "type": "heading",
        "level": 2,
        "content": """
L'Agent 3 est responsable de la synthèse et de la génération du rapport final. Ses fonctions principales sont:

- Agrégation des résultats des Agents 1 et 2
- Génération d'une synthèse globale des résultats
- Création d'un raisonnement économique complet
- Production de sections de discussion et de conclusion
- Génération de références bibliographiques
- Création d'un rapport PDF bien structuré et formaté
- Production optionnelle d'un document Word si les bibliothèques requises sont disponibles

L'Agent 3 utilise des templates Jinja2 pour la génération de HTML, convertit ce HTML en PDF avec WeasyPrint, et peut également générer des documents Word avec python-docx. Il assure la présentation finale des résultats dans un format académique professionnel."""
    },
    {
        "title": "3.5 Utilitaires LLM (llm_utils.py)",
        "type": "heading",
        "level": 2,
        "content": """
Le module llm_utils.py fournit une interface unifiée pour interagir avec différents modèles de langage. Ses fonctions principales sont:

- Support de deux backends: Ollama (local) et Gemini (API cloud)
- Gestion des appels aux modèles de langage avec ou sans images
- Gestion robuste des erreurs et des timeouts
- Parsing et validation des réponses
- Logging détaillé pour faciliter le débogage

Ce module est utilisé par tous les agents pour communiquer avec les modèles de langage, offrant une couche d'abstraction qui simplifie l'interaction avec différents backends."""
    },
    {
        "title": "4. Technologies et Dépendances",
        "type": "heading",
        "level": 1,
        "content": """
L'agent d'analyse économétrique repose sur plusieurs technologies et bibliothèques clés:

Analyse de données et visualisation:
- pandas: Pour la manipulation et l'analyse des données
- matplotlib/seaborn: Pour la génération de visualisations
- statsmodels: Pour les modèles économétriques et statistiques
- numpy: Pour les calculs numériques

Modèles de langage:
- Ollama: Interface pour les modèles de langage exécutés localement
- API Gemini de Google: Pour l'accès aux modèles de langage dans le cloud

Génération de rapports:
- Jinja2: Pour les templates HTML
- WeasyPrint: Pour la conversion HTML vers PDF
- python-docx (optionnel): Pour la génération de documents Word
- Markdown: Pour le formatage du texte

Utilitaires et infrastructure:
- logging: Pour la gestion des logs
- argparse: Pour le parsing des arguments de ligne de commande
- json: Pour la manipulation des données JSON
- subprocess: Pour l'exécution de commandes externes
- requests: Pour les appels API REST
- base64: Pour l'encodage des images

Le système est conçu pour fonctionner dans un environnement Python 3.x et peut utiliser différents modèles de langage selon les besoins et la disponibilité."""
    },
    {
        "title": "5. Paramètres et Configuration",
        "type": "heading",
        "level": 1,
        "content": """
L'agent d'analyse économétrique peut être configuré via plusieurs paramètres:

Paramètres de ligne de commande (pipeline.py):
- csv_file: Chemin vers le fichier CSV à analyser (obligatoire)
- user_prompt: Requête utilisateur décrivant l'analyse souhaitée (obligatoire)
- --model: Modèle LLM à utiliser (défaut: gemma3:27b)
- --backend: Backend LLM à utiliser ('ollama' ou 'gemini', défaut: 'ollama')
- --academic: Générer un rapport au format académique (activé par défaut)

Variables d'environnement:
- GEMINI_API_KEY: Clé API pour l'accès à Gemini (obligatoire si backend='gemini')
- GOOGLE_API_KEY: Alias pour GEMINI_API_KEY (défini automatiquement)
- PYTHONUNBUFFERED: Défini à "1" pour garantir un logging immédiat

Paramètres internes configurables:
- Timeouts pour les appels API (DEFAULT_GEMINI_TIMEOUT dans llm_utils.py)
- Modèles par défaut (DEFAULT_OLLAMA_MODEL, DEFAULT_GEMINI_MODEL dans llm_utils.py)
- Nombre maximal de tentatives pour la correction de code (max_attempts dans agent2.py)

Le système utilise également des fichiers de log spécifiques pour chaque composant:
- pipeline.log: Log principal du pipeline
- agent1.log: Log de l'Agent 1
- agent2.log: Log de l'Agent 2
- agent3.log: Log de l'Agent 3

Ces logs peuvent être utilisés pour le débogage et le suivi de l'exécution."""
    },
    {
        "title": "6. Limitations et Extensions Possibles",
        "type": "heading",
        "level": 1,
        "content": """
Limitations actuelles:

1. Dépendance aux modèles de langage: La qualité des analyses dépend fortement des capacités des modèles de langage utilisés.

2. Gestion des erreurs: Bien que le système tente de corriger automatiquement les erreurs, certaines situations complexes peuvent nécessiter une intervention manuelle.

3. Performance: L'exécution peut être lente, particulièrement avec des modèles locaux volumineux comme qwq:32b.

4. Validation des résultats: La validation des résultats économétriques reste limitée, et les interprétations peuvent parfois être trop générales.

5. Bibliothèques optionnelles: Certaines fonctionnalités (comme la génération de documents Word) dépendent de bibliothèques qui ne sont pas installées par défaut.

Extensions possibles:

1. Support de formats de données supplémentaires: Ajouter la prise en charge d'autres formats comme Excel, parquet, ou SQL.

2. Interface utilisateur: Développer une interface web pour faciliter l'utilisation du système.

3. Parallélisation: Optimiser les performances en exécutant certaines tâches en parallèle.

4. Validation et tests automatisés: Ajouter des mécanismes de validation plus robustes pour les résultats économétriques.

5. Extensions multilingues: Ajouter le support pour d'autres langues que le français.

6. Modèles spécialisés: Intégrer des modèles de langage spécifiquement fine-tunés pour l'analyse économétrique.

7. Intégration avec des outils existants: Connecter le système à des environnements comme Jupyter, R Studio, ou des plateformes de visualisation comme Tableau."""
    },
    {
        "title": "7. Code Source Complet",
        "type": "heading",
        "level": 1,
        "content": "Cette section présente le code source complet de chaque composant du système, avec des explications détaillées sur la structure et le fonctionnement du code."
    }
]

def create_pdf():
    """
    Crée le document PDF avec la documentation complète.
    """
    print(f"Génération du document PDF: {OUTPUT_PDF}")
    
    # Configurer le document
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=A4,
        rightMargin=2*cm,
        leftMargin=2*cm,
        topMargin=2*cm,
        bottomMargin=2*cm
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Style personnalisé pour le titre principal - modifier le style existant au lieu d'en ajouter un nouveau
    styles['Title'].fontSize = 20
    styles['Title'].spaceAfter = 24
    styles['Title'].alignment = TA_CENTER
    
    # Style personnalisé pour les titres de sections - modifier les styles existants
    styles['Heading1'].fontSize = 16
    styles['Heading1'].spaceAfter = 12
    styles['Heading1'].spaceBefore = 24
    
    # Style personnalisé pour les sous-titres
    styles['Heading2'].fontSize = 14
    styles['Heading2'].spaceAfter = 10
    styles['Heading2'].spaceBefore = 18
    
    # Style personnalisé pour le texte normal
    styles['BodyText'].fontSize = 11
    styles['BodyText'].leading = 14
    styles['BodyText'].alignment = TA_JUSTIFY
    
    # Style pour le code source
    if 'Code' in styles:
        styles['Code'].fontSize = 9
        styles['Code'].leading = 11
        styles['Code'].spaceBefore = 6
        styles['Code'].spaceAfter = 12
        styles['Code'].fontName = 'Courier'
    else:
        # Si le style Code n'existe pas, le créer
        styles.add(ParagraphStyle(
            name='Code',
            fontName='Courier',
            fontSize=9,
            leading=11,
            spaceBefore=6,
            spaceAfter=12
        ))
    
    # Éléments du document
    elements = []
    
    # Générer le contenu structuré
    for section in DOCUMENT_STRUCTURE:
        if section["type"] == "heading":
            if section["level"] == 0:
                elements.append(Paragraph(section["title"], styles["Title"]))
                elements.append(Spacer(1, 0.5*inch))
            elif section["level"] == 1:
                if elements:  # Ajouter un saut de page sauf pour la première section
                    elements.append(PageBreak())
                elements.append(Paragraph(section["title"], styles["Heading1"]))
                elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=6, spaceAfter=12))
            elif section["level"] == 2:
                elements.append(Paragraph(section["title"], styles["Heading2"]))
        
        if "content" in section and section["content"]:
            content = section["content"].strip()
            paragraphs = content.split("\n\n")
            for p in paragraphs:
                if p.strip():
                    elements.append(Paragraph(p.strip(), styles["BodyText"]))
                    elements.append(Spacer(1, 0.1*inch))
    
    # Ajouter le code source de chaque fichier
    for file in SOURCE_FILES:
        if os.path.exists(file):
            elements.append(PageBreak())
            elements.append(Paragraph(f"7.{SOURCE_FILES.index(file) + 1} Code Source: {file}", styles["Heading2"]))
            
            # Lire le contenu du fichier
            with open(file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # Analyser le code pour extraire la docstring principale
            docstring_match = re.search(r'"""(.*?)"""', code, re.DOTALL)
            if docstring_match:
                docstring = docstring_match.group(1).strip()
                elements.append(Paragraph("Description:", styles["BodyText"]))
                elements.append(Paragraph(docstring, styles["BodyText"]))
                elements.append(Spacer(1, 0.2*inch))
            
            # Analyser les fonctions et classes pour créer une structure
            functions = re.findall(r'def ([^\(]+)\([^\)]*\):\s*(?:"""(.*?)""")?', code, re.DOTALL)
            classes = re.findall(r'class ([^\(]+)(?:\([^\)]*\))?:\s*(?:"""(.*?)""")?', code, re.DOTALL)
            
            if functions or classes:
                elements.append(Paragraph("Structure du fichier:", styles["BodyText"]))
                
                if classes:
                    elements.append(Paragraph("Classes:", styles["BodyText"]))
                    for class_name, class_doc in classes:
                        elements.append(Paragraph(f"• {class_name}: {class_doc.strip().split('.')[0] if class_doc else 'Pas de description'}", styles["BodyText"]))
                
                if functions:
                    elements.append(Paragraph("Fonctions:", styles["BodyText"]))
                    for func_name, func_doc in functions:
                        elements.append(Paragraph(f"• {func_name}: {func_doc.strip().split('.')[0] if func_doc else 'Pas de description'}", styles["BodyText"]))
            
            elements.append(Spacer(1, 0.2*inch))
            
            # Ajouter le code source complet
            elements.append(Paragraph("Code source complet:", styles["BodyText"]))
            
            # Diviser le code en sections plus petites pour éviter les problèmes de mise en page
            code_lines = code.split('\n')
            chunk_size = 100  # Nombre de lignes par bloc
            for i in range(0, len(code_lines), chunk_size):
                chunk = '\n'.join(code_lines[i:i+chunk_size])
                elements.append(Preformatted(chunk, styles["Code"]))
                if i + chunk_size < len(code_lines):
                    elements.append(Spacer(1, 0.1*inch))
    
    # Générer le PDF
    doc.build(elements)
    print(f"Document PDF généré avec succès: {OUTPUT_PDF}")

if __name__ == "__main__":
    create_pdf()