# STREAMING_CHUNK:Initializing Streamlit app and configuration...
import os
import re
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

fallback_pdf_path = None

st.set_page_config(
    page_title="Agente RAG Corporativo", page_icon="🤖", layout="centered"
)

st.title("🤖 Asistente RAG Corporativo con Gemini")
st.markdown(
    "Sube tu documento PDF o usa el predeterminado para realizar consultas"
    " inteligentes."
)

# Configurar API Key
api_key = os.getenv("GEMINI_API_KEY")
try:
  if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
  pass

with st.sidebar:
  st.header("⚙️ Configuración")
  input_key = st.text_input(
      "Gemini API Key",
      value=api_key or "",
      type="password",
  )
  if input_key:
    api_key = input_key
    os.environ["GEMINI_API_KEY"] = api_key

  st.divider()
  st.subheader("📄 Cargar Documento")
  uploaded_file = st.file_uploader(
      "Sube tu PDF (ej. futureapplication.pdf)", type=["pdf"]
  )

if not api_key:
  st.warning("⚠️ Por favor, ingresa tu API Key de Gemini en la barra lateral.")
  st.stop()
else:
  os.environ["GEMINI_API_KEY"] = api_key

st.sidebar.markdown(
    "\n---\n" \
    "### Modelo Gemini\n" \
    "Puedes ajustar el modelo con la variable de entorno `GEMINI_CHAT_MODEL`."
)


# STREAMING_CHUNK:Processing uploaded PDF and generating vector store...
@st.cache_resource
def inicializar_rag_desde_archivo(file_bytes_io, filename):
  global fallback_pdf_path
  # Guardar temporalmente el archivo subido para que PyPDFLoader pueda leerlo
  with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(file_bytes_io.read())
    tmp_path = tmp_file.name
    fallback_pdf_path = tmp_path

  loader = PyPDFLoader(tmp_path)
  documentos = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=200
  )
  chunks = text_splitter.split_documents(documentos)

  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vector_store = FAISS.from_documents(chunks, embeddings)
  return vector_store.as_retriever(search_kwargs={"k": 3})


def _fallback_answer(query: str, error_text: str) -> str | None:
  try:
    import agent
    if fallback_pdf_path:
      agent.FILE_PATH = fallback_pdf_path
      agent.document_chunks = []
      agent.document_sentences = []
      agent.initialization_error = None
    return agent._keyword_fallback_answer(query, error_text)
  except Exception:
    return None


# Determinar qué archivo usar (el subido por el usuario o buscar localmente)
target_file = None
file_name_display = ""

if uploaded_file is not None:
  target_file = uploaded_file
  file_name_display = uploaded_file.name
elif os.path.exists("futureapplication.pdf"):
  target_file = open("futureapplication.pdf", "rb")
  file_name_display = "futureapplication.pdf"
  fallback_pdf_path = "futureapplication.pdf"

if target_file is None:
  st.info(
      "👈 Por favor, **sube tu archivo PDF** en la barra lateral para comenzar a"
      " chatear con el agente."
  )
  st.stop()

st.success(f"✅ Documento activo: **{file_name_display}**")

try:
  retriever = inicializar_rag_desde_archivo(target_file, file_name_display)
  model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
  llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
except Exception as e:
  st.error(f"Error procesando el PDF: {e}")
  st.stop()


# STREAMING_CHUNK:Rendering chat UI and message loop...
if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])

if query := st.chat_input("Escribe tu pregunta sobre el documento..."):
  st.session_state.messages.append({"role": "user", "content": query})
  with st.chat_message("user"):
    st.markdown(query)

  with st.chat_message("assistant"):
    with st.spinner("Buscando en el documento y generando respuesta..."):
      try:
        docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            "Eres un asistente inteligente experto en análisis de documentos"
            " corporativos.\nUsa los siguientes fragmentos de contexto para"
            " responder la pregunta en español.\nSi no sabes la respuesta, dilo"
            f" claramente.\n\nContexto:\n{context_text}\n\nPregunta: {query}"
        )
        response = llm.invoke(prompt)
        st.markdown(response.content)
        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )
      except Exception as e:
        fallback_text = _fallback_answer(query, None)
        if fallback_text:
          st.markdown(fallback_text)
          st.session_state.messages.append(
              {"role": "assistant", "content": fallback_text}
          )
        else:
          st.error("Ocurrió un error al procesar la consulta.")