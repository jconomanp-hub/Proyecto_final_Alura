# STREAMING_CHUNK:Initializing imports and environment...
import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Configuración de la API Key directamente en el código para pruebas locales
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
print("✅ API Key de Gemini configurada correctamente en el script.")

# 2. Configurar la ruta del documento
# STREAMING_CHUNK:Validating document path...
FILE_PATH = "futureapplication.pdf"  # Asegúrate de que tu PDF esté en la misma carpeta

if not os.path.exists(FILE_PATH):
  raise FileNotFoundError(
      f"❌ No se encontró el archivo '{FILE_PATH}' en esta carpeta."
      " Colócalo aquí antes de continuar."
  )

def cargar_documento(file_path):
  ext = file_path.split(".")[-1].lower()
  if ext == "pdf":
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"📖 PDF cargado. Páginas leídas: {len(documents)}")
  elif ext == "csv":
    loader = CSVLoader(file_path)
    documents = loader.load()
    print(f"📊 CSV cargado. Registros leídos: {len(documents)}")
  else:
    raise ValueError("Formato no soportado. Usa PDF o CSV.")
  return documents

print("🔄 Procesando documento...")
documentos_brutos = cargar_documento(FILE_PATH)

# 3. Segmentación en Chunks
# STREAMING_CHUNK:Splitting document text into chunks...
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
chunks = text_splitter.split_documents(documentos_brutos)
print(f"✂️ Documento fragmentado en {len(chunks)} bloques de texto.")

# 4. Crear Base de Datos Vectorial (FAISS) con el modelo de embeddings correcto
# STREAMING_CHUNK:Building vector database with gemini-embedding-001...
print("📦 Generando embeddings y base de datos vectorial (FAISS)...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

batch_size = 20
vector_store = None

for i in range(0, len(chunks), batch_size):
  batch = chunks[i : i + batch_size]
  print(
      f"Procesando lote {i // batch_size + 1} de"
      f" {(len(chunks) - 1) // batch_size + 1}..."
  )

  if vector_store is None:
    vector_store = FAISS.from_documents(batch, embeddings)
  else:
    vector_store.add_documents(batch)

  if i + batch_size < len(chunks):
    time.sleep(15)  # Pausa para respetar límites de la API de Google

print("✅ Base de datos vectorial creada exitosamente.")

# 5. Configurar el LLM con gemini-2.0-flash y la cadena RAG
# STREAMING_CHUNK:Configuring LLM and RAG chain...
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

system_prompt = (
    "Eres un asistente inteligente experto en análisis de documentos internos"
    " corporativos.\nUsa los siguientes fragmentos de contexto recuperados para"
    " responder la pregunta.\nRealiza la traducción en español para entregar la"
    " respuesta.\nSi no sabes la respuesta o no está en el documento, di"
    " claramente que no dispones de esa información.\nMantén las respuestas"
    " claras, concisas y profesionales.\n\nContexto:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain,
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Interfaz Interactiva en la Terminal con control de reintentos (503)
# STREAMING_CHUNK:Starting agent interactive loop with retry logic...
if __name__ == "__main__":
  print(
      "\n🤖 ¡Agente inteligente listo! Escribe tu pregunta o 'salir' para"
      " terminar.\n"
  )
  while True:
    pregunta = input("🙋‍♂️ Pregunta: ")
    if pregunta.lower() in ["salir", "exit", "quit"]:
      print("¡Hasta luego!")
      break

    # Reintento automático en caso de saturación temporal (Error 503)
    max_reintentos = 3
    intentos = 0
    exito = False

    while intentos < max_reintentos and not exito:
      try:
        respuesta = rag_chain.invoke({"input": pregunta})
        print(f"\n🤖 Respuesta del Agente:\n{respuesta['answer']}\n")
        print("-" * 60)
        exito = True
      except Exception as e:
        intentos += 1
        if "503" in str(e) and intentos < max_reintentos:
          print(
              f"⚠️ Servidores ocupados (503). Reintentando en 5 segundos"
              f" (Intento {intentos}/{max_reintentos})..."
          )
          time.sleep(5)
        else:
          print(f"❌ Ocurrió un error al procesar la pregunta: {e}")
          print("-" * 60)
        