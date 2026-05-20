import requests
from bs4 import BeautifulSoup
import time

class AmarnaMultiScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    def extraer_de_url(self, url):
        print(f"-> Analizando: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            if response.status_code != 200: return ""
            
            soup = BeautifulSoup(response.text, 'html.parser')

            # --- LIMPIEZA QUIRÚRGICA ---
            # Eliminamos todo lo que NO sea el artículo
            for junk in soup.select('.header, .nav, .footer, .mntl-header-nav, .mntl-side-list, .mntl-attribution-container'):
                junk.decompose()

            # El contenido real en esta web siempre está aquí:
            cuerpo = soup.find('div', {'id': 'article-body_1-0'}) or soup.find('article')
            
            if not cuerpo: return ""

            texto_acumulado = f"\n=== FUENTE: {url} ===\n"
            
            # Buscamos encabezados (Preguntas) y párrafos (Estrategias)
            for el in cuerpo.find_all(['h2', 'h3', 'p', 'li']):
                texto = el.get_text().strip()
                
                # Filtros de calidad
                if len(texto) < 25: continue
                if any(bad in texto.lower() for bad in ['privacy policy', 'view all', 'partners', 'advertisement']): continue
                
                if el.name in ['h2', 'h3']:
                    texto_acumulado += f"\n[PREGUNTA/TEMA]: {texto.upper()}\n"
                else:
                    texto_acumulado += f"• {texto}\n"
            
            return texto_acumulado

        except Exception as e:
            print(f"Error en {url}: {e}")
            return ""

    def ejecutar(self, urls):
        print("--- INICIANDO SCRAPEO MASIVO PARA CHATAMARNA ---")
        conocimiento_total = "--- BASE DE CONOCIMIENTO MAESTRA: PREGUNTAS Y RESPUESTAS ---"
        
        for url in urls:
            contenido = self.extraer_de_url(url)
            conocimiento_total += contenido
            time.sleep(2) # Pausa de cortesía para no ser bloqueados

        with open("rag_maestro_amarna.txt", "w", encoding="utf-8") as f:
            f.write(conocimiento_total)
        
        print("\n--- ¡ÉXITO TOTAL! ---")
        print("Archivo 'rag_maestro_amarna.txt' generado con el contexto de todas las webs.")

if __name__ == "__main__":
    urls_objetivo = [
        "https://www.thebalancemoney.com/problem-solving-interview-questions-and-answers-5083660",
        "https://www.thebalancemoney.com/what-to-say-in-a-job-interview-4158527",
        "https://www.thebalancemoney.com/questions-to-ask-during-your-job-interview-2071488",
        "https://www.thebalancemoney.com/interview-questions-about-strengths-and-job-performance-2064065",
        "https://www.thebalancemoney.com/job-interview-question-how-would-you-describe-yourself-2064058"
    ]
    
    scraper = AmarnaMultiScraper()
    scraper.ejecutar(urls_objetivo)