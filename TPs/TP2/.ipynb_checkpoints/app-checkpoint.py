import os
import tempfile

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# -----------------------------
# Configuración global / entorno
# -----------------------------
os.environ["ALLOW_CHROMA_TELEMETRY"] = "false"
os.environ["OPENAI_API_KEY"] = ""

# -----------------------------
# Configuración básica de la página
# -----------------------------
st.set_page_config(
    page_title="Chateá con tu CV",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Chateá con tu CV")
st.write(
    "Hacele preguntas al CV, compará contra Job Descriptions, analizá los fragmentos utilizados para las respuestas :)"
)

# -----------------------------
# Helpers
# -----------------------------


def load_docs_from_pdf(path: str):
    """Carga un PDF y lo transforma en documentos LangChain."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs


def create_rag_components(docs, model_name: str, temperature: float, api_key: str):
    """
    Crea los componentes del pipeline RAG:
    - splitter
    - embeddings + Chroma + retriever
    - LLM
    - Prompt
    """

    # 1) Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        add_start_index=True,
    )
    docs_chunked = text_splitter.split_documents(docs)

    # 2) Embeddings + vector store
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )
    vectordb = Chroma.from_documents(docs_chunked, embedding=embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # 3) LLM
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )

    # 4) Prompt para forzar uso del contexto
    template = """
Eres un asistente que responde preguntas sobre el CV de un alumno.

Debes responder **únicamente** usando la información del contexto
Si la respuesta no está en el contexto, responde exactamente:
"No tengo esa información en el CV."

📄 CONTEXTO:
{context}

❓ PREGUNTA:
{question}

🧠 RESPUESTA clara, en español y bien estructurada:
"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    return retriever, llm, prompt


def answer_question(question: str, retriever, llm, prompt: PromptTemplate):
    """Ejecuta un paso de RAG: retrieve → formatear prompt → llamar al modelo."""
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    return response.content, docs


# -----------------------------
# Sidebar: configuración y carga de CV
# -----------------------------
st.sidebar.header("⚙️ Configuración")

# API Key (puede venir de env o desde la UI)
env_api_key = os.getenv("OPENAI_API_KEY") or ""
api_key = st.sidebar.text_input(
    "🔑 OpenAI API Key",
    type="password",
    value=env_api_key,
    help="Tu clave de OpenAI. No se guarda en ningún lado.",
)

if not api_key:
    st.sidebar.warning("Ingresá tu OpenAI API Key para continuar.")
    st.stop()

# Modelo y temperatura
model_name = st.sidebar.selectbox(
    "Modelo",
    options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    index=0,
)
temperature = st.sidebar.slider("Creatividad (temperature)", 0.0, 1.0, 0.2, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 CV del alumno")

uploaded_file = st.sidebar.file_uploader(
    "Subí tu CV en PDF",
    type=["pdf"],
    help="Si no subís nada, se usa el archivo por defecto en 'CV Lucas Argento.pdf' en la misma carpeta.",
)

DEFAULT_CV_PATH = "CV Lucas Argento.pdf"

# -----------------------------
# Carga de documentos
# -----------------------------
docs = None

if uploaded_file is not None:
    # Guardamos el PDF subido en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    docs = load_docs_from_pdf(tmp_path)
else:
    # Usar el CV por defecto si existe
    if os.path.exists(DEFAULT_CV_PATH):
        docs = load_docs_from_pdf(DEFAULT_CV_PATH)
    else:
        st.error(
            "No se encontró un CV. Subí un PDF desde la barra lateral "
            "o crea el archivo 'CV Lucas Argento.pdf' en esta carpeta."
        )

if docs is None:
    st.stop()

# -----------------------------
# Crear / actualizar componentes RAG
# -----------------------------
if (
    "retriever" not in st.session_state
    or st.session_state.get("qa_model") != model_name
    or st.session_state.get("qa_temp") != temperature
    or st.session_state.get("docs_source") != ("uploaded" if uploaded_file else "default")
    or st.session_state.get("qa_api_key") != api_key
):
    retriever, llm, prompt = create_rag_components(docs, model_name, temperature, api_key)
    st.session_state.retriever = retriever
    st.session_state.llm = llm
    st.session_state.prompt = prompt
    st.session_state.qa_model = model_name
    st.session_state.qa_temp = temperature
    st.session_state.docs_source = "uploaded" if uploaded_file else "default"
    st.session_state.qa_api_key = api_key

retriever = st.session_state.retriever
llm = st.session_state.llm
prompt = st.session_state.prompt

# -----------------------------
# Historial de chat
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Mostrar historial previo
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -----------------------------
# Botón de demo automática
# -----------------------------
st.sidebar.markdown("---")
if st.sidebar.button("🔧 Correr demo automática"):
    demo_questions = [
        "¿Dónde estudia el alumno?",
        "¿Qué experiencia laboral tiene?",
        "¿Qué tecnologías o lenguajes de programación utiliza?",
    ]
    for q in demo_questions:
        with st.chat_message("user"):
            st.markdown(q)

        answer, _docs_used = answer_question(q, retriever, llm, prompt)

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.history.append({"role": "user", "content": q})
        st.session_state.history.append({"role": "assistant", "content": answer})

    st.stop()

# -----------------------------
# Input tipo chat
# -----------------------------
query = st.chat_input("Preguntame algo sobre el CV del alumno")

if query:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(query)

    # Ejecutar RAG
    answer, docs_used = answer_question(query, retriever, llm, prompt)

    # Mostrar respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Guardar en historial
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Mostrar contexto recuperado
    with st.expander("🔍 Ver fragmentos del CV usados para responder"):
        for i, d in enumerate(docs_used, start=1):
            st.markdown(f"**Fragmento {i}:**")
            st.write(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))
            st.markdown("---")

# Mensaje inicial si todavía no hay conversación
if not st.session_state.history and query is None:
    st.info(
        "Escribí una pregunta en el cuadro de chat de abajo. Por ejemplo:\n\n"
        "- ¿Dónde estudia el alumno?\n"
        "- ¿Qué experiencia laboral tiene?\n"
        "- ¿Qué lenguajes de programación usa?"
    )
