from PIL import Image
import numpy as np
import os
import json
import io
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import google.generativeai as genai
from pypdf import PdfReader


from doctr.io import DocumentFile
from doctr.models import ocr_predictor

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

import logging

# Configurar logging para depuración en archivo
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if not api_key:
    logging.error("No se encontró la GEMINI_API_KEY en el archivo .env")
    raise ValueError("¡ERROR CRÍTICO! No se encontró la GEMINI_API_KEY en el archivo .env")

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash", 
    generation_config={
        "temperature": 0.2, 
        "response_mime_type": "application/json"
    }
)

app = FastAPI(title="JobMatch Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Cargando modelo OCR...")
predictor_ocr = ocr_predictor(pretrained=True)
print("Modelo OCR cargado correctamente.")

class VacanteItem(BaseModel):
    id: str
    titulo: str
    empresa: str
    descripcion: str
    
class MatchResultado(BaseModel):
    vacante_id: str
    titulo: str
    porcentaje_match: int
    razon_clave: str = Field(description="Explicación persuasiva para el alumno")

def obtener_vacantes_db():
    return [
        {
            "id": "VAC-001",
            "titulo": "Junior Python Backend",
            "empresa": "TechStart S.L.",
            "descripcion": "Buscamos programador junior para API REST. Valoramos FastAPI y Docker. No necesaria experiencia previa.",
            "categoria": "informatica"
        },
        {
            "id": "VAC-002",
            "titulo": "Auxiliar de Marketing Digital",
            "empresa": "Agencia Creativa",
            "descripcion": "Gestión de redes y diseño básico. Se valora portfolio creativo y uso de herramientas de IA.",
            "categoria": "marketing"
        },
        {
            "id": "VAC-003",
            "titulo": "Desarrollador Fullstack Junior",
            "empresa": "Consultora IT",
            "descripcion": "Mantenimiento de aplicaciones web. Stack: React y Node.js. Trabajo presencial.",
            "categoria": "informatica"
        }
    ]

def extraer_texto_pdf(file_bytes):
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        texto = ""
        for page in reader.pages:
            texto += page.extract_text() or ""
        return texto
    except Exception:
        raise HTTPException(status_code=400, detail="El archivo está corrupto o no es un PDF válido.")

def extraer_texto_imagen(file_bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        temp_io = io.BytesIO()
        img.save(temp_io, format="PNG")
        clean_bytes = temp_io.getvalue()
        doc = DocumentFile.from_images([clean_bytes])
        resultado = predictor_ocr(doc)

        texto_extraido = ""
        for page in resultado.pages:
            for block in page.blocks:
                for line in block.lines:
                    linea_texto = " ".join([word.value for word in line.words])
                    texto_extraido += linea_texto + "\n"
        
        return texto_extraido
    except Exception as e:
        import traceback
        logging.error(f"Error procesando imagen OCR: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail="El archivo de imagen no es válido o está corrupto.")

@app.post("/analizar-cv")
async def analizar_cv_endpoint(file: UploadFile = File(...)):
    """
    Endpoint principal: Recibe PDF/PNG/JPG -> Extrae Texto -> Consulta Gemini -> Devuelve JSON
    """
    # Log para depuración
    print(f"Archivo recibido: {file.filename}, Content-Type: {file.content_type}")
    
    # Normalizar content_type y extensiones
    tipos_pdf = ["application/pdf", "application/x-pdf", "application/octet-stream"]
    tipos_img = ["image/png", "image/jpeg", "image/jpg", "image/webp"]
    
    # Aceptar si el content_type es válido O si la extensión lo es (por si acaso el móvil envía mal el mime)
    ext = file.filename.lower().split('.')[-1]
    es_pdf_ext = ext == 'pdf'
    es_img_ext = ext in ['png', 'jpg', 'jpeg', 'webp']

    if file.content_type not in (tipos_pdf + tipos_img) and not (es_pdf_ext or es_img_ext):
        raise HTTPException(
            status_code=400, 
            detail=f"Tipo de archivo no soportado ({file.content_type}). Solo se aceptan PDF, PNG o JPG"
        )
    
    content = await file.read()

    # Decidir qué procesador usar (PDF o Imagen)
    # Priorizamos la extensión del archivo porque los móviles a veces envían mimes genéricos
    if es_pdf_ext:
        texto_cv = extraer_texto_pdf(content)
    elif es_img_ext:
        texto_cv = extraer_texto_imagen(content)
    elif file.content_type in ["application/pdf", "application/x-pdf"]:
        texto_cv = extraer_texto_pdf(content)
    else:
        # Por defecto si ha pasado la validación previa, lo tratamos como imagen
        texto_cv = extraer_texto_imagen(content)

    if len(texto_cv.strip()) < 50:
        raise HTTPException(status_code=400, detail="El archivo parece estar vacío o no pudimos extraer suficiente texto legible.")

    vacantes = obtener_vacantes_db()

    prompt = f"""
    Actúa como un reclutador experto que busca conectar talento junior con empresas.
    
    Analiza este CURRICULUM:
    ---
    {texto_cv}
    ---

    Compáralo con estas VACANTES DISPONIBLES:
    ---
    {json.dumps(vacantes)}
    ---

    TU TAREA:
    Devuelve un JSON con las vacantes que tengan un match SUPERIOR al 40%.
    
    REGLAS DE NEGOCIO:
    1. Si es perfil DAM/DAW, busca coincidencias técnicas (lenguajes, frameworks).
    2. Si es Marketing, busca herramientas y soft skills.
    3. IMPORTANTE: El campo 'razon_clave' debe ser una frase dirigida al alumno explicándole por qué encaja. 
        Ejemplo: "Tu proyecto de GitHub demuestra que sabes usar FastAPI, igual que piden aquí."
    
    Estructura de salida esperada:
    [
      {{ "vacante_id": "...", "titulo": "...", "porcentaje_match": 85, "razon_clave": "..." }}
    ]
    """

    try:
        response = model.generate_content(prompt)
        resultado = json.loads(response.text)
        # Return both the match results and the full extracted text for the app to store
        return {
            "results": resultado,
            "original_text": texto_cv
        }
    except Exception as e:
        import traceback
        logging.error(f"Error interno Gemini/JSON: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error procesando la IA. Inténtalo de nuevo.")

class ChatMessage(BaseModel):
    role: str # 'user' or 'model'
    content: str

class ChatRequest(BaseModel):
    message: str
    context: str = ""
    history: List[ChatMessage] = []
    is_final: bool = False

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Chatbot Agent: Entrenador IA personalizado con perfil de alumno.
    """
    system_instruction = f"""
Eres el asistente de selección de personal y entrenador de entrevistas de la App Amarna.
Tu objetivo es ayudar al alumno a prepararse para entrevistas técnicas y profesionales.

CONTEXTO DEL ALUMNO (Currículum/Perfil):
---
{request.context if request.context else "Candidato sin CV proporcionado. Pregunta por su experiencia general."}
---

INSTRUCCIONES DE COMPORTAMIENTO:
1. Responde siempre basándote en esta información del alumno cuando sea relevante.
2. Haz preguntas de entrevista desafiantes pero constructivas.
3. Si el alumno responde a una pregunta, dale feedback breve y pasa a la siguiente.
4. Mantén un tono profesional, motivador y experto en reclutamiento IT/Marketing.
5. Si te preguntan algo fuera de este contexto profesional, redirige amablemente la conversación a su carrera.

Responde de forma concisa y directa.
    """

    # We rebuild the history into the format expected by Google GenAI (list of parts or chat session)
    # For simplicity with the existing 'model' object, we use generate_content with a full prompt sequence
    # or better, start a chat session.
    
    if request.is_final:
        system_instruction += """
\n¡ATENCIÓN! El usuario ha respondido por última vez. La entrevista ha terminado.
Debes evaluar toda la entrevista basándote en su CV y las respuestas.
Devuelve tu evaluación ESTRICTAMENTE en este formato JSON:
{
  "comunicacion": 85,
  "tecnologia": 80,
  "feedback_general": "Tus puntos fuertes y áreas de mejora...",
  "mensaje_despedida": "¡Gracias por tu tiempo! Aquí tienes tus resultados..."
}
"""

    try:
        # Re-configuring model with system instruction for this specific call 
        chat_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction,
            generation_config={"response_mime_type": "application/json"} if request.is_final else {}
        )
        
        # Format history for Gemini chat session
        formatted_history = []
        for msg in request.history:
            formatted_history.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [msg.content]
            })
            
        chat_session = chat_model.start_chat(history=formatted_history)
        response = chat_session.send_message(request.message)
        
        return {"response": response.text}
        
    except Exception as e:
        print(f"Error en Chatbot: {e}")
        raise HTTPException(status_code=500, detail=f"Error en el Chatbot: {str(e)}")


# ─────────────────────────────────────────────────────────────
# ENDPOINT: Transcripción de Audio con Gemini
# ─────────────────────────────────────────────────────────────
@app.post("/transcribir-audio")
async def transcribir_audio(file: UploadFile = File(...)):
    """
    Recibe un archivo de audio (m4a, mp3, wav, ogg…) y lo transcribe
    usando el modelo Gemini, que entiende audio de forma nativa.
    """
    import tempfile, pathlib

    # Extensiones MIME aceptadas por Gemini para audio
    mime_map = {
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".aac": "audio/aac",
        ".webm": "audio/webm",
    }

    suffix = pathlib.Path(file.filename or "audio.m4a").suffix.lower() or ".m4a"
    mime_type = mime_map.get(suffix, "audio/mp4")

    try:
        audio_bytes = await file.read()

        # Guardar en un fichero temporal para poder subirlo
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Subir el audio a la Files API de Gemini (hasta 20 MB)
        import google.generativeai as genai_files
        uploaded = genai_files.upload_file(
            path=tmp_path,
            mime_type=mime_type,
            display_name="amarna_audio_transcripcion"
        )

        # Esperar a que el archivo esté procesado
        import time
        max_wait = 30  # segundos máximo
        waited = 0
        while uploaded.state.name == "PROCESSING" and waited < max_wait:
            time.sleep(1)
            waited += 1
            uploaded = genai_files.get_file(uploaded.name)

        if uploaded.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="El archivo de audio falló al procesarse en Gemini.")

        # Modelo para transcripción (texto plano, sin JSON)
        transcription_model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            generation_config={"temperature": 0.0}
        )

        response = transcription_model.generate_content([
            uploaded,
            "Por favor, transcribe exactamente el habla en este audio al español. "
            "Devuelve SOLO el texto transcrito, sin puntuación adicional ni comentarios."
        ])

        # Limpiar fichero temporal
        import os as _os
        try:
            _os.unlink(tmp_path)
        except Exception:
            pass

        return {"text": response.text.strip()}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error en transcripción de audio: {e}")
        logging.error(f"Error en transcripción de audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error al transcribir audio: {str(e)}")

