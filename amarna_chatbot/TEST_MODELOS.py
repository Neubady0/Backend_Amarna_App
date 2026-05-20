import os
from google import genai

LLAVE = os.environ.get("GEMINI_API_KEY", "AIzaSyBijeCToGUgK6y8M9znLL6TDJQo6zrGMyw")
client = genai.Client(api_key=LLAVE)

print("--- VERIFICANDO CONEXIÓN CON AMARNA ---")
try:
    modelos = list(client.models.list())
    for m in modelos:
        print(f"-> Modelo disponible: {m.name}")
except Exception as e:
    print(f"Error de conexión: {e}")