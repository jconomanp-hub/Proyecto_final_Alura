import json
import os
import re
import urllib.parse
import urllib.request
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()

FILE_PATH = os.getenv("PDF_PATH", "futureapplication.pdf")
api_key = os.getenv("GOOGLE_API_KEY")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
STOPWORDS = {
    "a",
    "al",
    "de",
    "del",
    "la",
    "las",
    "el",
    "los",
    "un",
    "una",
    "y",
    "o",
    "para",
    "que",
    "como",
    "en",
    "es",
    "son",
    "por",
    "con",
    "su",
    "sus",
    "se",
    "si",
    "no",
    "esta",
    "este",
    "esta",
    "está",
    "están",
    "el",
    "lo",
    "la",
    "les",
    "the",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "on",
    "is",
    "are",
    "this",
    "that",
    "what",
    "why",
    "how",
    "when",
    "where",
}


class State(TypedDict):
    messages: list


retriever = None
llm = None
initialization_error = None
document_chunks = []
document_sentences = []


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).strip()


def _load_local_documents():
    global document_chunks, document_sentences, initialization_error

    if document_chunks and document_sentences:
        return True

    if document_chunks and not document_sentences:
        document_sentences[:] = []
        for chunk in document_chunks:
            document_sentences.extend(_split_sentences(chunk))
        return True

    if not os.path.exists(FILE_PATH):
        initialization_error = f"No se encontró el archivo '{FILE_PATH}'."
        return False

    try:
        loader = PyPDFLoader(FILE_PATH)
        documentos = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(documentos)
        document_chunks[:] = [chunk.page_content for chunk in chunks]
        document_sentences[:] = []
        for chunk in document_chunks:
            document_sentences.extend(_split_sentences(chunk))
        return True
    except Exception as exc:
        initialization_error = str(exc)
        return False


def _split_sentences(text: str) -> list[str]:
    sentence_boundaries = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in sentence_boundaries if part.strip()]


def _translate_to_spanish_external(text: str) -> str:
    try:
        encoded = urllib.parse.quote_plus(text)
        url = (
            "https://translate.googleapis.com/translate_a/single?client=gtx"
            "&sl=auto&tl=es&dt=t&q=" + encoded
        )
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8")
        payload = json.loads(body)
        translated = "".join(part[0] for part in payload[0] if part and part[0])
        return translated.strip()
    except Exception:
        return ""


def _is_heading_or_toc(sentence: str) -> bool:
    text = sentence.strip()
    if not text:
        return True
    if text.count(".") >= 3 and len(text) < 80:
        return True
    if re.fullmatch(r"[\d\.\s\-:]+", text):
        return True
    if "..." in text and len(text) < 120:
        return True
    if len(text.split()) <= 4 and any(char.isdigit() for char in text):
        return True
    if text.lower() in {"índice", "tabla de contenidos", "contenido", "capítulo", "introducción", "resumen"}:
        return True
    return False


def _paragraphs_from_chunk(chunk: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", chunk) if p.strip()]
    return paragraphs if paragraphs else [chunk]


def _is_toc_chunk(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False

    toc_lines = 0
    for line in lines:
        lower = line.lower()
        if _is_heading_or_toc(line):
            toc_lines += 1
            continue
        if re.search(r"\b(página|page|pg|p\.)\b", lower) and re.search(r"\d{1,4}", lower):
            toc_lines += 1

    return toc_lines >= max(3, int(len(lines) * 0.35))


def _strip_toc_lines(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if re.fullmatch(r"[\d\.\s\-:]+", stripped):
            continue
        if stripped.count(".") >= 3 and len(stripped) < 80:
            continue
        if "..." in stripped and len(stripped) < 120:
            continue
        if any(kw in lower for kw in ["índice", "tabla de contenidos", "contenido", "capítulo", "capitulo", "sección", "seccion"]):
            continue
        if re.search(r"\b(página|page)\s*\d+\b", lower):
            continue
        if lower.endswith(tuple(str(i) for i in range(1, 1000))):
            no_num = re.sub(r"\d+$", "", stripped).strip()
            if len(no_num) > 0 and len(no_num.split()) <= 7:
                continue
        lines.append(stripped)
    return " ".join(lines)


def _extract_relevant_sentence(query: str) -> str:
    if not _load_local_documents():
        return ""

    normalized_query = _normalize_text(query)
    keywords = [word for word in normalized_query.split() if word and word not in STOPWORDS]
    if not keywords:
        keywords = normalized_query.split()

    candidates = []
    for chunk in document_chunks:
        chunk_text = _normalize_text(chunk)
        score = 0
        for keyword in keywords:
            occurrences = chunk_text.count(keyword)
            score += occurrences * 3
        if any(term in chunk_text for term in keywords):
            score += 1
        if len(chunk_text.split()) < 30:
            score -= 5
        if _is_toc_chunk(chunk):
            score -= 100
        candidates.append((score, chunk))

    candidates.sort(key=lambda item: item[0], reverse=True)

    for score, best_chunk in candidates:
        if score < 2:
            break
        if _is_toc_chunk(best_chunk):
            continue
        cleaned_chunk = _strip_toc_lines(best_chunk)
        paragraphs = _paragraphs_from_chunk(cleaned_chunk)
        best_paragraph = ""
        best_paragraph_score = -1
        for paragraph in paragraphs:
            if _is_heading_or_toc(paragraph):
                continue
            normalized_paragraph = _normalize_text(paragraph)
            paragraph_score = sum(normalized_paragraph.count(keyword) for keyword in keywords)
            if paragraph_score > best_paragraph_score:
                best_paragraph_score = paragraph_score
                best_paragraph = paragraph
        if best_paragraph:
            return best_paragraph
        if paragraphs:
            return paragraphs[0]

    sentence_candidates = []
    for sentence in document_sentences:
        if _is_heading_or_toc(sentence):
            continue
        normalized_sentence = _normalize_text(sentence)
        score = sum(1 for keyword in keywords if keyword in normalized_sentence)
        if score > 0:
            sentence_candidates.append((score, sentence))

    if sentence_candidates:
        sentence_candidates.sort(key=lambda item: item[0], reverse=True)
        return sentence_candidates[0][1]

    for sentence in document_sentences:
        if not _is_heading_or_toc(sentence):
            return sentence

    return ""


def _keyword_fallback_answer(query: str, error: str | None = None) -> str:
    best_sentence = _extract_relevant_sentence(query)

    if best_sentence:
        # Try to translate the selected sentence into Spanish using the LLM
        translated = None
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI  # re-import safe
            if _initialize_llm_only():
                translate_prompt = (
                    "Traduce al español el siguiente texto manteniendo el significado y la terminología técnica cuando aplique. Responde únicamente con la traducción en español:\n\n" + best_sentence
                )
                resp = llm.invoke(translate_prompt)
                translated = getattr(resp, "content", str(resp))
        except Exception:
            translated = None

        if translated:
            answer = f"Según el documento, {translated}"
        else:
            external = _translate_to_spanish_external(best_sentence)
            if external:
                answer = f"Según el documento, {external}"
            else:
                answer = f"Según el documento, {best_sentence}"
    else:
        answer = (
            "No pude encontrar una respuesta clara en el documento local."
        )

    if error:
        answer += f" Detalles: {error}"
    return answer


def _initialize_resources():
    global retriever, llm, initialization_error

    if retriever is not None and llm is not None:
        return True

    initialization_error = None

    if not _load_local_documents():
        return False

    if not api_key:
        initialization_error = "No se encontró GOOGLE_API_KEY en el entorno."
        return False

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
        vector_store = FAISS.from_documents(
            [chunk for chunk in document_chunks if chunk],
            embeddings,
        )
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
        return True
    except Exception as exc:
        initialization_error = str(exc)
        return False


def _translate_to_spanish_external(text: str) -> str:
    try:
        encoded = urllib.parse.quote_plus(text)
        url = (
            "https://translate.googleapis.com/translate_a/single?client=gtx"
            "&sl=auto&tl=es&dt=t&q=" + encoded
        )
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8")
        payload = json.loads(body)
        translated = "".join(part[0] for part in payload[0] if part and part[0])
        return translated.strip()
    except Exception:
        return ""


def _initialize_llm_only() -> bool:
    """Initialize only the LLM for lightweight tasks like translation.

    This avoids creating embeddings or FAISS indexes and helps when the
    embedding quota is exhausted but the LLM can still be used for translation.
    """
    global llm, initialization_error

    if llm is not None:
        return True

    if not api_key:
        initialization_error = "No se encontró GOOGLE_API_KEY en el entorno."
        return False

    try:
        llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.0)
        return True
    except Exception as exc:
        initialization_error = str(exc)
        return False


def call_model(state: State):
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, dict):
        query = last_message.get("content", "")
    else:
        query = getattr(last_message, "content", str(last_message))

    if not _initialize_resources():
        fallback_message = _keyword_fallback_answer(query)
        return {"messages": [AIMessage(content=fallback_message)]}

    try:
        docs = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in docs])

        prompt = (
            "Eres un asistente inteligente experto en análisis de documentos internos"
            " corporativos.\nResponde siempre en español y, si es necesario, traduce"
            " la respuesta al español.\nUsa los siguientes fragmentos de contexto para"
            " responder la pregunta.\nSi no sabes la respuesta o no está en el"
            " documento, di claramente que no dispones de esa información.\n\nContexto:\n"
            f"{context_text}\n\nPregunta: {query}"
        )

        response = llm.invoke(prompt)
        generated = getattr(response, "content", str(response))
        translated = _translate_to_spanish_external(generated)
        if translated:
            return {"messages": [AIMessage(content=translated)]}
        return {"messages": [AIMessage(content=generated)]}
    except Exception as exc:
        fallback_message = _keyword_fallback_answer(query, str(exc))
        return {"messages": [AIMessage(content=fallback_message)]}


workflow = StateGraph(State)
workflow.add_node("rag_agent", call_model)
workflow.add_edge(START, "rag_agent")
workflow.add_edge("rag_agent", END)

graph = workflow.compile()