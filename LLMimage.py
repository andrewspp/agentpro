import base64
import requests
import json
import time

# Début du chronomètre
print("Début du test:", time.strftime("%H:%M:%S"))

# Charger l'image
image_path = "/Users/pierreandrews/Desktop/AgentPro/education_revenu_relation.png"
with open(image_path, "rb") as f:
    image_data = f.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

print(f"Image chargée: {len(image_data)} octets")

# Clé API
api_key = "AIzaSyAYT-NjrJiRK9Ei8gp716uR57CO59puWhg"

# Appel direct à l'API REST de Gemini
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
headers = {"Content-Type": "application/json"}

# Construction de la requête
payload = {
    "contents": [
        {
            "parts": [
                {"text": "Analyse cette visualisation économique en détail."},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_base64
                    }
                }
            ]
        }
    ],
    "generationConfig": {
        "temperature": 0.4,
        "maxOutputTokens": 800,
        "topP": 0.8
    }
}

print(f"Envoi de la requête API à {url}...")
try:
    # Définir un timeout explicite
    response = requests.post(url, headers=headers, json=payload, timeout=30)
    print(f"Statut de la réponse: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print("Réponse reçue avec succès!")
        if 'candidates' in data and len(data['candidates']) > 0:
            text = data['candidates'][0]['content']['parts'][0]['text']
            print("\n=== DESCRIPTION DE L'IMAGE ===\n")
            print(text)
        else:
            print(f"Structure de réponse inattendue: {json.dumps(data, indent=2)}")
    else:
        print(f"Erreur API: {response.text}")
except requests.exceptions.Timeout:
    print("La requête a expiré (timeout après 30 secondes)")
except requests.exceptions.RequestException as e:
    print(f"Erreur lors de la requête HTTP: {e}")
except Exception as e:
    print(f"Erreur inattendue: {e}")

print("Fin du test:", time.strftime("%H:%M:%S"))