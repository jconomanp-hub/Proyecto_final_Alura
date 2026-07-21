# 🤖 Agente de Inteligencia Artificial para Consulta de Documentos (RAG)

Solución basada en Inteligencia Artificial para consulta de documentos corporativos mediante lenguaje natural, con soporte local y traducción a español.

---

## 📌 Resumen del Proyecto

Este proyecto implementa un asistente RAG (Retrieval-Augmented Generation) con:
- `langgraph` para ejecutar el flujo de consulta local.
- `langchain` y `langchain-google-genai` para interactuar con Google Gemini.
- `FAISS` para indexar texto extraído de PDF.
- `PyPDFLoader` para cargar documentos PDF.
- Fallback local basado en búsqueda de texto y traducción de contenido.

El objetivo es responder preguntas usando contenido real del documento y evitar respuestas basadas solo en índices o tablas de contenido.

---

## 🧠 Arquitectura y Comportamiento

1. **Carga de documentos**
   - Se lee `futureapplication.pdf` (o el archivo definido en `PDF_PATH`).
   - El texto se divide en chunks mediante `RecursiveCharacterTextSplitter`.

2. **RAG con Gemini**
   - Si hay `GOOGLE_API_KEY`, se construye un índice FAISS con embeddings de `GoogleGenerativeAIEmbeddings`.
   - Se usa `ChatGoogleGenerativeAI` para generar respuestas en español.

3. **Fallback local seguro**
   - Si la inicialización de embeddings o el servidor de Gemini falla, usa búsqueda local en los chunks del PDF.
   - Selecciona el chunk más relevante según coincidencias de palabras clave y evita chunks de tabla de contenido.
   - Extrae el párrafo más relevante del chunk encontrado.
   - Traduce el resultado al español usando primero Gemini y, si falla, el endpoint público `translate.googleapis.com`.

4. **Mejoras clave implementadas**
   - `agent.py` ahora realiza inicialización perezosa para evitar errores al importar el módulo.
   - Se evita devolver texto de tabla de contenido o índices.
   - Se prioriza contenido real del PDF y respuestas más largas y útiles.

---

## ⚙️ Requisitos y Variables de Entorno

Variables de entorno recomendadas:

- `GOOGLE_API_KEY` → Clave para usar Gemini y embeddings.
- `PDF_PATH` → Ruta del PDF a procesar (por defecto `futureapplication.pdf`).
- `GEMINI_CHAT_MODEL` → Modelo de chat, por ejemplo `gemini-2.5-flash`.
- `GEMINI_EMBEDDING_MODEL` → Modelo de embeddings, por ejemplo `gemini-embedding-001`.
- `GEMINI_API_KEY` → Clave de Gemini usada por `app_streamlit.py` cuando se ejecuta la interfaz Streamlit.

Ejemplo `.env`:

```env
GOOGLE_API_KEY=tu_api_key
PDF_PATH=futureapplication.pdf
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_API_KEY=tu_api_key_gemini
```

### Configuración segura en Streamlit

El archivo `app_streamlit.py` usa la clave de Gemini de manera segura desde:
- `GEMINI_API_KEY` en variables de entorno,
- `st.secrets["GEMINI_API_KEY"]`,
- o ingreso manual en la barra lateral de Streamlit.

No se debe dejar la API key embebida en el código fuente.

---

## 🧪 Pruebas

Se agregó cobertura de pruebas para:
- importar `agent.py` sin inicializar la RAG completa.
- devolver fallback local cuando Gemini no está disponible.
- usar la lógica de búsqueda local en contenido real.
- evitar que el fallback devuelva solo tabla de contenido.

Ejecutar pruebas:

```bash
c:/Users/JuanCoñomanPeralta/proyecto_alura/venv/Scripts/python.exe -m unittest discover -s tests
```

---

## 📁 Estructura del Proyecto

- `agent.py` → Define el grafo de `langgraph`, la lógica RAG y el fallback local.
- `app.py` → (opcional) punto de entrada adicional para la aplicación.
- `app_streamlit.py` → Interfaz Streamlit para consulta interactiva.
- `README.md` → Documentación del proyecto.
- `requirements.txt` → Dependencias del proyecto.
- `tests/test_agent.py` → Pruebas unitarias del fallback y la inicialización.
- `futureapplication.pdf` → Documento fuente para la consulta.

---

## 🚀 Cómo Usar

1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```

2. Coloca el PDF en la raíz del proyecto o actualiza `PDF_PATH`.
3. Crea `.env` con tu `GOOGLE_API_KEY`.
4. Ejecuta el servidor de `langgraph` o el módulo Python.

Si `langgraph` está instalado:

```bash
langgraph dev
```

Para ejecutar la interfaz Streamlit:

```bash
streamlit run app_streamlit.py
```

Asegúrate de tener definida `GEMINI_API_KEY` en el entorno o en los secretos de Streamlit.

Para usar el agente directamente desde Python:

```python
from agent import call_model

state = {"messages": [{"content": "¿Cómo reemplazo una OTU?"}]}
print(call_model(state))
```

---

## 📌 Notas Importantes

- El fallback local está diseñado para ser robusto cuando el servicio de Gemini no está disponible.
- El agente siempre intenta responder en español y mantiene la terminología técnica cuando es posible.
- Si no existe `GOOGLE_API_KEY`, el proyecto sigue funcionando con fallback local, pero sin la generación basada en Gemini.

---

## 🔧 Cambios recientes

- Implementación de **búsqueda local de texto** cuando la RAG completa no está disponible.
- Filtrado de resultados para **evitar tablas de contenido** e información de índice.
- Traducción a español con Gemini y respaldo en el endpoint público de Google Translate.
- Test de regresión que valida el comportamiento del fallback y la selección de contenido.
