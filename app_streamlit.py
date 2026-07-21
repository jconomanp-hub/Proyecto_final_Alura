# STREAMING_CHUNK:Initializing Streamlit web interface...
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

# Configurar API Key de forma segura desde los Secrets de Streamlit o input
api_key = st.secrets.get("GEMINI_API_KEY")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
else:
    st.warning("No se encontró GEMINI_API_KEY en los secrets. Define la clave en Streamlit o usa una variable de entorno externa.")


@st.cache_resource
def inicializar_rag():
  # Cargar documento y embeddings locales
  loader = PyPDFLoader("futureapplication.pdf")
  documentos = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000, chunk_overlap=200
  )
  chunks = text_splitter.split_documents(documentos)

  embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
  vector_store = FAISS.from_documents(chunks, embeddings)
  return vector_store.as_retriever(search_kwargs={"k": 3})


retriever = inicializar_rag()
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

# Historial de chat en Streamlit
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