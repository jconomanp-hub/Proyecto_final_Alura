# STREAMING_CHUNK:Initializing Streamlit app and configuration...
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="Agente RAG Corporativo", page_icon="🤖", layout="centered"
)

st.title("🤖 Asistente RAG Corporativo con Gemini")
st.markdown("Consulta inteligente sobre `futureapplication.pdf`")

# Configurar API Key desde Streamlit Secrets, variable de entorno o la barra lateral
api_key = os.getenv("GEMINI_API_KEY")
try:
  if not api_key and "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
  pass

# Barra lateral para ingresar la clave si no está definida
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

if not api_key:
  st.warning(
      "⚠️ Por favor, ingresa tu API Key de Gemini en la barra lateral o define"
      " la variable de entorno GEMINI_API_KEY."
  )
  st.stop()
else:
  os.environ["GEMINI_API_KEY"] = api_key

# STREAMING_CHUNK:Loading PDF document and setting up retriever...
@st.cache_resource
def inicializar_rag():
  # Verificar si el PDF existe localmente
  pdf_path = "futureapplication.pdf"
  if not os.path.exists(pdf_path):
    st.error(
        f"❌ No se encontró el archivo '{pdf_path}' en el repositorio de GitHub."
        " Asegúrate de subirlo."
    )
    st.stop()

  loader = PyPDFLoader(pdf_path)
  documentos = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=200
  )
  chunks = text_splitter.split_documents(documentos)

  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vector_store = FAISS.from_documents(chunks, embeddings)
  return vector_store.as_retriever(search_kwargs={"k": 3})


try:
  retriever = inicializar_rag()
  llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
except Exception as e:
  st.error(f"Error inicializando el RAG: {e}")
  st.stop()

# STREAMING_CHUNK:Rendering chat interface and conversation history...
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
        error_msg = f"Ocurrió un error al procesar la consulta: {e}"
        st.error(error_msg)