import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

from PIL import Image
import numpy as np
import os
import json
import io
import math
import re
import logging
import time
import requests
import traceback
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from gestor_rag import BaseDatosVectorialAmarna, ElementoRAG
from google import genai
from google.genai import types
from pypdf import PdfReader
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

ruta_env = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ruta_env, override=True)

logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("CRITICO: No se encontro GEMINI_API_KEY en .env")

client = genai.Client(api_key=api_key)
app = FastAPI(title="Amarna Elite Logistics & Match API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

predictor_ocr = ocr_predictor(pretrained=True)
rag_db = BaseDatosVectorialAmarna(api_key=api_key)

ruta_json_barrios = Path(__file__).parent / "barrios_bcn.json"
DATASET_BARRIOS = {}
if ruta_json_barrios.exists():
    with open(ruta_json_barrios, "r", encoding="utf-8") as f:
        DATASET_BARRIOS = json.load(f)

class DatosEstructuradosCV(BaseModel):
    codigo_postal: str = Field(description="Código postal de 5 dígitos extraído del CV. Si no aparece, usar '08001'")
    perfil_resumido: str = Field(description="Resumen de las habilidades técnicas y experiencia encontradas")

class DatosEmpresaPro(BaseModel):
    nombre_empresa: str
    reputacion_y_cultura: str = Field(description="Analisis del ambiente real de trabajo y metodologias utilizadas")
    puntos_clave_entrevista: List[str] = Field(description="Consejos sobre que remarcar en la entrevista")

class MatchOfertaAvanzado(BaseModel):
    oferta_id: str
    titulo_puesto: str
    empresa: str
    porcentaje_accuracy: int = Field(description="Nivel de encaje tecnico y logistico entre 0 y 100")
    distancia_km: float = Field(description="Distancia calculada desde el barrio/CP del candidato")
    tiempo_trayecto_estimado: str = Field(description="Mensaje personalizado sobre el desplazamiento nombrando los barrios exactos")
    justificacion_match: str = Field(description="Explicacion detallada del encaje")
    investigacion_corporativa: DatosEmpresaPro

class PasoRoadmap(BaseModel):
    tecnologia_faltante: str
    enlace_recurso: str = Field(description="URL real recomendada de estudio")
    accion_requerida: str = Field(description="Instruccion directa de dedicacion temporal")

class PropuestaMejora(BaseModel):
    objetivo: str = Field(description="Aplicable a 'CV', 'GitHub' o 'Portfolio'")
    sugerencia: str = Field(description="Enfoque a modificar para maximizar el impacto")
    ejemplo_redaccion: str = Field(description="Mapeo practico mostrando el Antes y el Despues")

class RespuestaEliteAmarna(BaseModel):
    top_3_ofertas: List[MatchOfertaAvanzado]
    carta_presentacion_infiltrada: str = Field(description="Carta de presentacion organica para la oferta #1")
    roadmap_cierra_gaps: List[PasoRoadmap] = Field(description="Plan enfocado en suplir carencias tecnicas")
    plan_mejoras_perfil: List[PropuestaMejora]

def parsear_json_seguro(texto: str):
    texto = texto.strip()
    if texto.startswith("```json"):
        texto = texto[7:]
    elif texto.startswith("```"):
        texto = texto[3:]
    if texto.endswith("```"):
        texto = texto[:-3]
    return json.loads(texto.strip())

def interceptar_cp_seguridad(texto_cv: str, cp_ia: str) -> str:
    match = re.search(r'\b(08\d{3})\b', texto_cv)
    if match:
        return match.group(1)
    return cp_ia

def resolver_datos_cp(cp: str):
    if cp in DATASET_BARRIOS:
        data = DATASET_BARRIOS[cp]
        return (data["lat"], data["lng"]), data["barrio"], cp
    for cp_k, data in DATASET_BARRIOS.items():
        if cp[:4] == cp_k[:4]:
            return (data["lat"], data["lng"]), data["barrio"], cp
        if cp[:3] == cp_k[:3]:
            return (data["lat"], data["lng"]), data["barrio"], cp
    return (41.3874, 2.1686), "Área Metropolitana BCN", cp

def calcular_haversine(lat1, lon1, lat2, lon2) -> float:
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 15.0  
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 1)

def extraer_readme_github(url: str) -> str:
    if not url or "github.com" not in url:
        return ""
    try:
        base = url.replace("github.com", "raw.githubusercontent.com").rstrip('/')
        for rama in ["master", "main"]:
            res = requests.get(f"{base}/{rama}/README.md", timeout=5)
            if res.status_code == 200:
                return f"Repositorio {url}:\n{res.text[:2500]}"
    except Exception:
        pass
    return f"Enlace aportado: {url}"

MODELOS_FLASH = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"]

def generar_con_retry(contents, config, modelos=None, max_intentos=3):
    """Intenta generar contenido con retry y fallback de modelos ante 503/429."""
    if modelos is None:
        modelos = MODELOS_FLASH
    ultimo_error = None
    for modelo in modelos:
        for intento in range(max_intentos):
            try:
                return client.models.generate_content(model=modelo, contents=contents, config=config)
            except Exception as e:
                ultimo_error = e
                err_str = str(e)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    espera = 2 ** intento  # 1s, 2s, 4s
                    logging.warning(f"Modelo {modelo} intento {intento+1} falló ({err_str[:80]}). Reintentando en {espera}s...")
                    time.sleep(espera)
                else:
                    break  # Error no transitorio, probar siguiente modelo
    raise ultimo_error

def investigar_reputacion_empresa(empresa: str, puesto: str) -> str:
    if empresa.lower() in ["general", "empresa local", "empresa real"]:
        return "Cultura orientada a resultados con metodologias agiles estandar."
    prompt = f"Busca opiniones en foros IT o Glassdoor sobre trabajar en '{empresa}' en España como '{puesto}'. Resume su ambiente laboral y metodologias."
    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())], temperature=0.2)
        )
        return res.text
    except Exception:
        return "Empresa consolidada con enfoque en el desarrollo continuo."

@app.post("/analizar-cv", response_model=RespuestaEliteAmarna)
async def analizar_cv_endpoint(
    file: UploadFile = File(...), 
    github_url: Optional[str] = Form(None),
    lat_movil: Optional[float] = Form(None),
    lng_movil: Optional[float] = Form(None)
):
    ext = file.filename.lower().split('.')[-1]
    es_pdf = ext == 'pdf' or file.content_type in ["application/pdf", "application/x-pdf"]
    content = await file.read()
    texto_cv = PdfReader(io.BytesIO(content)).pages[0].extract_text() if es_pdf else extraer_texto_imagen(content)

    if len(texto_cv.strip()) < 50:
        raise HTTPException(status_code=400, detail="Documento ilegible.")

    texto_github = extraer_readme_github(github_url) if github_url else "Sin repositorio aportado."

    if lat_movil is not None and lng_movil is not None:
        lat_user, lng_user = lat_movil, lng_movil
        min_dist = float('inf')
        barrio_user = "Ubicación Actual"
        cp_user = "08001"
        for cp_k, data in DATASET_BARRIOS.items():
            d = calcular_haversine(lat_user, lng_user, data["lat"], data["lng"])
            if d < min_dist:
                min_dist = d
                barrio_user = data["barrio"]
                cp_user = cp_k
        origen_ubicacion = f"GPS Móvil ({barrio_user})"
    else:
        prompt_extraccion = f"Analiza este texto extraído de un currículum y conviértelo a JSON estrictamente. Presta especial atención a extraer el código postal de 5 dígitos (ej. 08030). CV: {texto_cv[:2000]}"
        try:
            res_cv_json = generar_con_retry(
                contents=prompt_extraccion,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json", 
                    response_schema=DatosEstructuradosCV, 
                    temperature=0.0
                )
            )
            datos_cv = parsear_json_seguro(res_cv_json.text)
            cp_ia = datos_cv.get("codigo_postal", "08001")
        except Exception:
            cp_ia = "08001"

        cp_user = interceptar_cp_seguridad(texto_cv, cp_ia)
        (lat_user, lng_user), barrio_user, cp_user = resolver_datos_cp(cp_user)
        origen_ubicacion = f"Texto CV (CP: {cp_user}, {barrio_user})"

    perfil_unificado = f"CV:\n{texto_cv}\n\nGitHub:\n{texto_github}"
    ofertas_candidatas = rag_db.buscar_candidatos(perfil_unificado, tipo="oferta", limite=15)

    if not ofertas_candidatas:
        raise HTTPException(status_code=404, detail="No se encontraron vacantes en el RAG.")

    candidatos_con_distancia = []
    for of in ofertas_candidatas:
        m = re.search(r'Código Postal:\s*(08\d{3})', of.contenido)
        cp_of = m.group(1) if m else "08001"
        
        (lat_of, lng_of), barrio_of, _ = resolver_datos_cp(cp_of)
        dist = calcular_haversine(lat_user, lng_user, lat_of, lng_of)
        
        candidatos_con_distancia.append({
            "oferta": of, 
            "distancia": dist, 
            "barrio_vacante": barrio_of,
            "cp_vacante": cp_of
        })

    candidatos_con_distancia.sort(key=lambda x: x["distancia"])
    top_3_seleccionadas = candidatos_con_distancia[:3]

    contexto_enriquecido = []
    for item in top_3_seleccionadas:
        of = item["oferta"]
        dist = item["distancia"]
        barrio_of = item["barrio_vacante"]
        info_cultura = investigar_reputacion_empresa(of.empresa, of.puesto)
        contexto_enriquecido.append({
            "id": of.id,
            "titulo": of.titulo,
            "empresa": of.empresa,
            "puesto": of.puesto,
            "barrio_vacante": barrio_of,
            "cp_vacante": item["cp_vacante"],
            "distancia_km": dist,
            "descripcion_tecnica": of.contenido,
            "investigacion_foros_cultura": info_cultura
        })

    prompt_maestro = f"Eres el motor de Inteligencia Artificial de Amarna. Diseña la respuesta de evaluacion de elite. DATOS LOGISTICOS DEL CANDIDATO: Origen de la ubicación: {origen_ubicacion}. PERFIL TECNICO: {perfil_unificado}. VACANTES SELECCIONADAS: {json.dumps(contexto_enriquecido, ensure_ascii=False)}. INSTRUCCIONES DE REDACCION PARA CUMPLIR EL ESQUEMA RespuestaEliteAmarna: 1. 'tiempo_trayecto_estimado': Redacta una estimacion de desplazamiento nombrando de forma explícita el barrio de residencia del candidato y el barrio de la vacante. 2. 'investigacion_corporativa': Extrae las metodologias reales del campo 'investigacion_foros_cultura'. 3. 'carta_presentacion_infiltrada': Redacta una carta de presentacion organica y persuasiva para la oferta 1. 4. 'roadmap_cierra_gaps': Incluye enlaces oficiales de aprendizaje y tiempos de dedicacion para suplir carencias."

    try:
        res_json = generar_con_retry(
            contents=prompt_maestro,
            config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=RespuestaEliteAmarna, temperature=0.2)
        )
        return parsear_json_seguro(res_json.text)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error interno durante la consolidacion del informe de IA: {str(e)}")

def extraer_texto_imagen(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    temp_io = io.BytesIO()
    img.save(temp_io, format="PNG")
    doc = DocumentFile.from_images([temp_io.getvalue()])
    res = predictor_ocr(doc)
    return "\n".join([" ".join([w.value for w in l.words]) for p in res.pages for b in p.blocks for l in b.lines])

@app.get("/admin/ver-memoria")
async def ver_memoria_rag(tipo_filtro: Optional[str] = None):
    if not rag_db.elementos:
        return {"total": 0, "elementos": []}
    resultados = rag_db.elementos
    if tipo_filtro:
        resultados = [el for el in resultados if el.tipo == tipo_filtro]
    return {"total": len(resultados), "elementos": [el.model_dump() for el in resultados]}

@app.delete("/admin/vaciar-rag")
async def vaciar_rag_endpoint():
    rag_db.limpiar_toda_la_memoria()
    return {"status": "success", "mensaje": "Base de datos vectorial eliminada con exito."}

@app.delete("/admin/eliminar-vacante/{vacante_id}")
async def eliminar_vacante_endpoint(vacante_id: str):
    exito = rag_db.eliminar_elemento_por_id(vacante_id)
    if not exito:
        raise HTTPException(status_code=404, detail="No se encontro la vacante.")
    return {"status": "success", "mensaje": "Vacante eliminada correctamente."}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)