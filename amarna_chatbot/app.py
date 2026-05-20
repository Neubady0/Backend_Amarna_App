from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from google import genai
from google.genai import errors
import os
import sqlite3
import json

app = Flask(__name__)
CORS(app)

# --- CONFIGURACIÓN GLOBAL ---
LLAVE = os.environ.get("GEMINI_API_KEY", "AIzaSyBijeCToGUgK6y8M9znLL6TDJQo6zrGMyw")
MODELO_ACTIVO = "models/gemini-2.5-flash"
client = genai.Client(api_key=LLAVE)

ESTRUCTURA_ENTREVISTA = """
Tu nombre es ChatAmarna. Debes dirigir la entrevista en este orden:
1. BLOQUE PERSONAL | 2. BLOQUE PROYECTO | 3. BLOQUE EMPRESA | 4. BLOQUE TRAMPA.
Regla de Oro: Evalúa si es momento de avanzar de bloque tras cada respuesta.
Si es el inicio, lanza la primera pregunta del Bloque Personal directamente.
"""

PERSONALIDADES = {
    "amable": (
        f"{ESTRUCTURA_ENTREVISTA}\n"
        "ROL: Mentor empático. TONO: Cálido y humano. "
        "ESTRATEGIA: Ayuda al candidato a brillar con preguntas guía."
    ),
    "agresivo": (
        f"{ESTRUCTURA_ENTREVISTA}\n"
        "ROL: Headhunter de élite. TONO: Hostil, seco y escéptico. "
        "ESTRATEGIA: Presiona al candidato, busca grietas y no tolera divagaciones."
    ),
    "tecnico": (
        f"{ESTRUCTURA_ENTREVISTA}\n"
        "ROL: Lead Developer senior. TONO: Analítico y neutral. "
        "ESTRATEGIA: Evalúa lógica, escalabilidad y profundidad técnica."
    )
}

# --- 1. PERSISTENCIA CON SQLITE ---
def init_db():
    conn = sqlite3.connect('amarna_sessions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS sessions 
                 (user_id TEXT PRIMARY KEY, history TEXT, summary TEXT)''')
    conn.commit()
    conn.close()

init_db()

def get_session_data(user_id):
    conn = sqlite3.connect('amarna_sessions.db')
    c = conn.cursor()
    c.execute("SELECT history, summary FROM sessions WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return json.loads(row[0]), row[1]
    return [], ""

def save_session_data(user_id, history, summary=""):
    conn = sqlite3.connect('amarna_sessions.db')
    c = conn.cursor()
    # Filtramos el historial para guardar solo texto (ahorra espacio y evita errores de binarios)
    clean_history = []
    for msg in history:
        if msg.get('parts') and 'text' in msg['parts'][0]:
            clean_history.append(msg)
    
    history_json = json.dumps(clean_history)
    c.execute("INSERT OR REPLACE INTO sessions VALUES (?, ?, ?)", (user_id, history_json, summary))
    conn.commit()
    conn.close()

# --- 2. RESUMEN DE CONTEXTO ---
def summarize_if_needed(user_id, history):
    if len(history) > 12: # Resumimos un poco antes para optimizar tokens
        try:
            prompt = f"Resume los puntos clave de esta entrevista de forma muy breve: {str(history)}"
            res = client.models.generate_content(model=MODELO_ACTIVO, contents=prompt)
            new_summary = res.text
            short_history = history[-4:] # Mantenemos solo los últimos 2 turnos
            save_session_data(user_id, short_history, new_summary)
        except: pass

# --- 3. RUTA PARA SUBIR EL RAG (CONEXIÓN) ---
@app.route('/api/upload_rag', methods=['POST'])
def upload_rag():
    file_path = "rag_maestro_amarna.txt"
    if not os.path.exists(file_path):
        return jsonify({'error': 'No existe el archivo. Ejecuta el scraper primero.'}), 400
    try:
        # Sube el archivo a los servidores de Google para que la IA lo procese
        uploaded_file = client.files.upload(path=file_path)
        return jsonify({'file_uri': uploaded_file.uri, 'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 4. MOTOR DE CHAT CON LÍMITES Y RAG ---
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = data.get('user_id', 'usuario_anonimo')
    user_input = data.get('message', '')
    modo = data.get('modo', 'amable').lower()
    datos_cv = data.get('cv_data', 'No hay CV cargado')
    file_uri = data.get('file_uri') # Recibe el URI del archivo RAG

    try:
        history_db, summary = get_session_data(user_id)
        instruccion = PERSONALIDADES.get(modo, PERSONALIDADES["amable"])
        sys_prompt = f"{instruccion}\nRESUMEN PREVIO: {summary}\nDATOS CV: {datos_cv}"

        # Preparar historial para Gemini
        history_to_send = []
        # Si es el inicio y hay RAG, lo inyectamos como primer mensaje de contexto
        if not history_db and file_uri:
            history_to_send.append({
                "role": "user",
                "parts": [{"file_data": {"file_uri": file_uri, "mime_type": "text/plain"}}]
            })
        history_to_send.extend(history_db)

        chat_session = client.chats.create(
            model=MODELO_ACTIVO,
            config={
                'system_instruction': sys_prompt,
                'temperature': 0.7,
                'max_output_tokens': 350, # LÍMITE DE TOKENS POR RESPUESTA
            },
            history=history_to_send
        )

        def generate():
            try:
                full_reply = ""
                for chunk in chat_session.send_message_stream(user_input):
                    full_reply += chunk.text
                    yield chunk.text
                
                # Actualizar historial local
                updated_history = []
                for msg in chat_session.get_history():
                    # No guardamos datos de archivos en el SQLite, solo el texto
                    if hasattr(msg.parts[0], 'text') and msg.parts[0].text:
                        updated_history.append({
                            "role": msg.role,
                            "parts": [{"text": msg.parts[0].text}]
                        })
                
                save_session_data(user_id, updated_history, summary)
                summarize_if_needed(user_id, updated_history)

            except errors.ClientError as ce:
                if "429" in str(ce) or "RESOURCE_EXHAUSTED" in str(ce):
                    yield "El chat se encuentra ocupado, prueba mas tarde."

        return Response(stream_with_context(generate()), mimetype='text/plain')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- 5. REPORTES Y FEEDBACK ---
@app.route('/api/report', methods=['POST'])
def get_report():
    data = request.get_json()
    user_id = data.get('user_id')
    history, _ = get_session_data(user_id)
    
    prompt = f"Analiza la entrevista {str(history)}. Devuelve JSON: hard_skills, soft_skills, cultural_fit, veredicto."
    try:
        response = client.models.generate_content(model=MODELO_ACTIVO, contents=prompt)
        json_clean = response.text.replace('```json', '').replace('```', '').strip()
        return jsonify(json.loads(json_clean))
    except:
        return jsonify({'error': 'Error en reporte'}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    user_id = request.get_json().get('user_id')
    conn = sqlite3.connect('amarna_sessions.db')
    c = conn.cursor()
    c.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)