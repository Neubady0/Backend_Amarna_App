import os
import json
import numpy as np
import faiss
from typing import List, Optional
from pydantic import BaseModel
from google import genai

class ElementoRAG(BaseModel):
    id: str
    tipo: str  
    titulo: str
    contenido: str
    empresa: Optional[str] = "General"
    puesto: Optional[str] = "Desarrollador"
    ubicacion: Optional[str] = "Barcelona"
    lat: Optional[float] = None
    lng: Optional[float] = None

class BaseDatosVectorialAmarna:
    def __init__(self, api_key: str, index_file="amarna_memoria.index", data_file="amarna_datos.json"):
        self.client = genai.Client(api_key=api_key)
        self.dimension = 3072
        self.index_file = index_file
        self.data_file = data_file
        self.elementos: List[ElementoRAG] = []
        self.index = None
        self.cargar_base_datos()

    def _generar_embedding(self, texto: str) -> np.ndarray:
        try:
            res = self.client.models.embed_content(
                model="gemini-embedding-2", 
                contents=texto
            )
            return np.array(res.embeddings[0].values, dtype=np.float32)
        except Exception as e1:
            try:
                res = self.client.models.embed_content(
                    model="gemini-embedding-001", 
                    contents=texto
                )
                return np.array(res.embeddings[0].values, dtype=np.float32)
            except Exception as e2:
                print(f"Fallo en embedding vectorial: {e1} | {e2}")
                return np.zeros(self.dimension, dtype=np.float32)

    def guardar_base_datos(self):
        if self.index is not None:
            faiss.write_index(self.index, self.index_file)
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump([e.model_dump() for e in self.elementos], f, ensure_ascii=False, indent=2)

    def cargar_base_datos(self):
        if os.path.exists(self.index_file) and os.path.exists(self.data_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    self.elementos = [ElementoRAG(**item) for item in json.load(f)]
            except Exception:
                self._init_vacio()
        else:
            self._init_vacio()

    def _init_vacio(self):
        self.index = faiss.IndexFlatL2(self.dimension)
        self.elementos = []

    def agregar_elementos(self, nuevos: List[ElementoRAG]):
        if not nuevos:
            return
        vectores = []
        for el in nuevos:
            texto = f"[{el.tipo.upper()}] Título: {el.titulo}. Empresa: {el.empresa}. Ubicación: {el.ubicacion}. Contenido: {el.contenido}"
            vectores.append(self._generar_embedding(texto))
            self.elementos.append(el)
        self.index.add(np.vstack(vectores))
        self.guardar_base_datos()

    def buscar_candidatos(self, consulta: str, tipo: str = "oferta", limite: int = 15) -> List[ElementoRAG]:
        if self.index is None or self.index.ntotal == 0:
            return []
        vec = np.expand_dims(self._generar_embedding(consulta), axis=0)
        distancias, indices = self.index.search(vec, limite * 2)
        
        resultados = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.elementos):
                item = self.elementos[idx]
                if item.tipo == tipo:
                    resultados.append(item)
                    if len(resultados) == limite:
                        break
        return resultados

    def limpiar_toda_la_memoria(self):
        self._init_vacio()
        self.guardar_base_datos()
        print("Memoria RAG completamente reseteada.")

    def eliminar_elemento_por_id(self, id_eliminar: str) -> bool:
        if self.index is None or self.index.ntotal == 0:
            return False
        idx_destino = -1
        for i, el in enumerate(self.elementos):
            if el.id == id_eliminar:
                idx_destino = i
                break
        if idx_destino == -1:
            return False
        vectores_actuales = [self.index.reconstruct(i) for i in range(self.index.ntotal)]
        self.elementos.pop(idx_destino)
        vectores_actuales.pop(idx_destino)
        self.index = faiss.IndexFlatL2(self.dimension)
        if vectores_actuales:
            self.index.add(np.array(vectores_actuales, dtype=np.float32))
        self.guardar_base_datos()
        return True