import os
import requests
import traceback  # To log full errors
from dotenv import load_dotenv
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

# Load environment variables from .env file
load_dotenv()

# Watsonx credentials
API_KEY = os.getenv("IBM_CLOUD_API_KEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
BASE_URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "ibm/granite-13b-instruct-v2"

# NLU credentials
NLU_API_KEY = os.getenv("NLU_API_KEY")
NLU_URL = os.getenv("NLU_URL")  # Use the raw URL from the service credential (no /v1/analyze)


# === Function to get Watsonx IAM token ===
def get_iam_token():
    try:
        token_url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = f"apikey={API_KEY}&grant_type=urn:ibm:params:oauth:grant-type:apikey"

        response = requests.post(token_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json()["access_token"]

    except Exception as e:
        print("Error getting IAM token:", e)
        return None


# === Granite Model Text Generation ===
def get_granite_response(prompt):
    if not API_KEY or not PROJECT_ID:
        return "ERROR: Missing Watsonx API credentials."

    token = get_iam_token()
    if not token:
        return "ERROR: Failed to get IAM access token."

    url = f"{BASE_URL}/ml/v1/text/generation?version=2024-05-01"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "input": prompt,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 200
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result['results'][0]['generated_text'].strip()

    except requests.exceptions.HTTPError as e:
        print("HTTPError:", e.response.text)
        return f"ERROR: {e.response.text}"
    except Exception as e:
        print("Error:", e)
        return f"ERROR: {e}"


# === Sentiment Analysis using Watson NLU ===
def get_sentiment(text):
    if not NLU_API_KEY or not NLU_URL:
        print("[DEBUG] Missing NLU credentials.")
        return {"label": "unavailable", "score": 0}

    try:
        print(f"[DEBUG] Analyzing text: {text}")
        print(f"[DEBUG] Using NLU URL: {NLU_URL}")

        authenticator = IAMAuthenticator(NLU_API_KEY)
        nlu = NaturalLanguageUnderstandingV1(
            version='2022-04-07',
            authenticator=authenticator
        )
        nlu.set_service_url(NLU_URL)

        response = nlu.analyze(
            text=text,
            features=Features(sentiment=SentimentOptions())
        ).get_result()

        print(f"[DEBUG] NLU response: {response}")
        return response['sentiment']['document']

    except Exception as e:
        print(f"[ERROR] in get_sentiment: {e}")
        traceback.print_exc()
        return {"label": "error", "score": 0}
