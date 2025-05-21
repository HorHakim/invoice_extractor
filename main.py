import streamlit as st
from mistralai import Mistral
import base64
import os
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Récupérer la clé API
api_key = os.getenv("MISTRAL_KEY")

# Initialiser le client Mistral
client = Mistral(api_key=api_key)

# Titre de l'application
st.title("Extraction automatique d'informations de facture")

# Upload de l'image
uploaded_image = st.file_uploader("Chargez votre facture (JPG ou PNG)", type=["jpg", "jpeg", "png"])

# Fonction pour encoder l'image en base64
def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# Traitement lorsque l'image est chargée
if uploaded_image:
    image_bytes = uploaded_image.read()
    base64_image = encode_image(image_bytes)

    st.image(image_bytes, caption="Image chargée", use_container_width =True)
    context = """
    Tu es un assistant très précis spécialisé dans l'extraction de données depuis des images de factures. Analyse attentivement la facture fournie et extrais uniquement les trois informations suivantes :

    Montant total (le montant total TTC facturé, incluant clairement la devise utilisée).

    Date de la facture (la date d'émission ou date inscrite explicitement comme date de facture au format JJ/MM/AAAA).

    Nom du fournisseur (le nom complet du fournisseur tel qu'il est explicitement mentionné en haut de la facture).

    Date d'échéance (au format JJ/MM/AAAA))

    Sois extrêmement attentif et précis, particulièrement si le texte est manuscrit, car certaines lettres ou chiffres pourraient être difficiles à interpréter. Si une donnée n'est pas clairement lisible ou absente, précise explicitement : "Information non lisible" ou "Information absente".

    Ne fournis aucune information supplémentaire ni de commentaire superflu. Ta réponse doit strictement respecter le format 
    """
    prompt = """
    Extrais clairement les informations suivantes de cette facture si elles sont disponibles :
    - Montant total
    - Date de la facture
    - Nom du fournisseur
    - Date d'échéance

    Fournis la réponse sous la forme suivante :
    Montant total : <valeur>
    Date de la facture : <valeur>
    Nom du fournisseur : <valeur>
    Date d'échéance : <valeur>
    """

    # Construction des messages pour Pixtral
    messages = [
        {
            "role" : "system",
            "content" : context
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
            ]
        }
    ]

    # Appel au modèle Pixtral
    with st.spinner("Extraction des données..."):
        try:
            response = client.chat.complete(
                model="pixtral-12b-2409",
                messages=messages
            )

            result = response.choices[0].message.content

            # Affichage des résultats
            st.subheader("Informations extraites :")
            st.write(result)

        except Exception as e:
            st.error(f"Une erreur s'est produite : {e}")
