import os
import time
import re
import json
import requests
from bs4 import BeautifulSoup
from pathlib import Path

ruta_env = Path(__file__).parent / ".env"
from dotenv import load_dotenv
load_dotenv(dotenv_path=ruta_env, override=True)

from gestor_rag import BaseDatosVectorialAmarna, ElementoRAG

ruta_json_barrios = Path(__file__).parent / "barrios_bcn.json"
DATASET_BARRIOS = {}
if ruta_json_barrios.exists():
    with open(ruta_json_barrios, "r", encoding="utf-8") as f:
        DATASET_BARRIOS = json.load(f)

def limpiar_municipio(ubicacion_str: str) -> str:
    if not ubicacion_str:
        return "Barcelona"
    parte = ubicacion_str.split(",")[0].strip()
    coletillas = ["y alrededores", "Area", "Area,", "Metropolitan", "Área metropolitana de", "Provincia de", "Greater"]
    for c in coletillas:
        if c in parte:
            parte = parte.replace(c, "").strip()
    if not parte or parte.lower() in ["españa", "spain", "catalonia", "cataluña"]:
        return "Barcelona"
    return parte

def deducir_cp_y_coordenadas(descripcion: str, ciudad_limpia: str):
    match = re.search(r'\b(08\d{3})\b', descripcion)
    if match:
        cp = match.group(1)
        if cp in DATASET_BARRIOS:
            return cp, DATASET_BARRIOS[cp]["lat"], DATASET_BARRIOS[cp]["lng"], DATASET_BARRIOS[cp]["barrio"]

    ciudad_target = ciudad_limpia.lower()
    if ciudad_target == "barcelona":
        return "08001", 41.3810, 2.1685, "El Raval (Ciutat Vella)"

    for cp, datos in DATASET_BARRIOS.items():
        if ciudad_target in datos["barrio"].lower():
            return cp, datos["lat"], datos["lng"], datos["barrio"]

    for cp, datos in DATASET_BARRIOS.items():
        barrio_lower = datos["barrio"].lower()
        if any(p in barrio_lower for p in ciudad_target.split() if len(p) > 3):
            return cp, datos["lat"], datos["lng"], datos["barrio"]

    return "08001", 41.3810, 2.1685, f"Zona {ciudad_limpia}"

def scrapear_barcelona_cfgs(rag_db: BaseDatosVectorialAmarna, nombre_rama: str, busquedas: list, total_por_rama: int = 25):
    print("\n" + "="*60)
    print(f"RECOLECTANDO OFERTAS EN BARCELONA PARA: {nombre_rama.upper()}")
    print("="*60)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }

    ids_ofertas = []
    for palabra_clave in busquedas:
        if len(ids_ofertas) >= total_por_rama:
            break
        print(f"\nBuscando vacantes locales de '{palabra_clave}'...")
        for start in range(0, 75, 25):
            url_search = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={palabra_clave}&location=Barcelona%2C%20Catalonia%2C%20Spain&start={start}"
            try:
                res = requests.get(url_search, headers=headers, timeout=10)
                if res.status_code != 200:
                    time.sleep(2)
                    continue
                soup = BeautifulSoup(res.text, "html.parser")
                for t in soup.find_all("li"):
                    card = t.find("div", {"class": "base-card"})
                    if card and card.get("data-entity-urn"):
                        jid = card.get("data-entity-urn").split(":")[3]
                        if jid not in ids_ofertas:
                            ids_ofertas.append(jid)
                time.sleep(1.5)
            except Exception:
                pass
            if len(ids_ofertas) >= total_por_rama:
                break

    print(f"Consolidando {len(ids_ofertas[:total_por_rama])} puestos. Descargando descripciones...")

    nuevos_elementos = []
    for idx, job_id in enumerate(ids_ofertas[:total_por_rama], 1):
        url_detalle = f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
        print(f"   -- [{idx}/{total_por_rama}] Extrayendo ID: {job_id}...")
        try:
            res_job = requests.get(url_detalle, headers=headers, timeout=10)
            if res_job.status_code != 200:
                time.sleep(1.5)
                continue
            job_soup = BeautifulSoup(res_job.text, "html.parser")
            
            h2 = job_soup.find("h2")
            titulo = h2.text.strip() if h2 else "Puesto Técnico"
            
            org = job_soup.find("a", class_=lambda c: c and "topcard__org-name-link" in c)
            if not org:
                org = job_soup.find("div", class_="top-card-layout__card")
            empresa = org.text.strip() if org else "Empresa Barcelona"

            loc_span = job_soup.find("span", class_=lambda c: c and "topcard__flavor--bullet" in c)
            ubicacion_bruta = loc_span.text.strip() if loc_span else "Barcelona"
            
            if "United States" in ubicacion_bruta or "EE.UU." in ubicacion_bruta:
                continue

            ciudad_limpia = limpiar_municipio(ubicacion_bruta)
            
            desc_div = job_soup.find("div", class_="show-more-less-html__markup")
            descripcion = desc_div.get_text(separator="\n", strip=True) if desc_div else ""
            
            cp_final, lat, lng, nombre_barrio = deducir_cp_y_coordenadas(descripcion, ciudad_limpia)
            
            if len(descripcion) > 100:
                contenido_enriquecido = f"Rama de Formación: CFGS {nombre_rama}\nBarrio Asignado: {nombre_barrio}\nCódigo Postal: {cp_final}\n\nDescripción:\n{descripcion[:3500]}"
                elemento = ElementoRAG(
                    id=f"LN-{job_id}",
                    tipo="oferta",
                    titulo=f"[{nombre_rama}] {titulo}",
                    empresa=empresa,
                    puesto=titulo,
                    ubicacion=nombre_barrio,
                    contenido=contenido_enriquecido,
                    lat=lat,
                    lng=lng
                )
                nuevos_elementos.append(elemento)
            time.sleep(1.2)
        except Exception:
            pass

    if nuevos_elementos:
        rag_db.agregar_elementos(nuevos_elementos)
        print(f"Indexadas {len(nuevos_elementos)} ofertas locales de {nombre_rama}.")

if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("CRITICO: Falta GEMINI_API_KEY en .env")

    rag_db = BaseDatosVectorialAmarna(api_key=api_key)

    config_busquedas = [
        {"rama": "DAM", "keywords": ["Programador DAM", "Desarrollador Flutter", "Mobile Developer"], "cantidad": 25},
        {"rama": "DAW", "keywords": ["Programador DAW", "Desarrollador Web", "Frontend React"], "cantidad": 25},
        {"rama": "ASIX", "keywords": ["Técnico ASIX", "Sistemas Informáticos", "Soporte IT"], "cantidad": 25},
        {"rama": "Marketing", "keywords": ["Marketing Digital", "Social Media Specialist", "SEO SEM"], "cantidad": 25}
    ]

    for config in config_busquedas:
        scrapear_barcelona_cfgs(rag_db=rag_db, nombre_rama=config["rama"], busquedas=config["keywords"], total_por_rama=config["cantidad"])
    print("\nBASE DE DATOS RAG POBLADA CON ÉXITO.")