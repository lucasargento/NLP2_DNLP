import os
import tempfile
import json
from typing import List, Dict, Optional, Tuple

# Configurar directorio temporal ANTES de importar librerías pesadas
# Esto soluciona problemas con PyTorch/sentence-transformers que necesitan TMPDIR
if not os.environ.get("TMPDIR"):
    # Intentar usar directorios temporales estándar
    _temp_dirs_to_try = ["/tmp", "/var/tmp", os.path.expanduser("~/tmp")]
    
    for tmp_dir in _temp_dirs_to_try:
        if os.path.exists(tmp_dir) and os.access(tmp_dir, os.W_OK):
            os.environ["TMPDIR"] = tmp_dir
            break
    else:
        # Crear directorio temporal local si no existe ninguno
        try:
            local_tmp = os.path.join(os.getcwd(), ".tmp")
            os.makedirs(local_tmp, exist_ok=True)
            if os.access(local_tmp, os.W_OK):
                os.environ["TMPDIR"] = local_tmp
            else:
                # Último recurso: usar el directorio actual
                os.environ["TMPDIR"] = os.getcwd()
        except Exception:
            # Si todo falla, usar el directorio actual
            os.environ["TMPDIR"] = os.getcwd()

import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

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
    page_title="Sistema Multi-Agente de CVs",
    page_icon="🤖",
    layout="wide",
)

st.title("Chateá con tu CV 2.0")
st.write(
    "Carga hasta 3 CVs de integrantes del equipo. El sistema detecta automáticamente sobre quién preguntas y enruta a los agentes correspondientes."
)

# -----------------------------
# Clase PersonAgent
# -----------------------------


class PersonAgent:
    """Agente RAG individual para una persona específica."""

    def __init__(
        self,
        name: str,
        docs: List[Document],
        model_name: str,
        temperature: float,
        api_key: str,
    ):
        self.name = name
        self.docs = docs

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
        # Crear colección única para cada agente usando el nombre
        # Limpiar el nombre para que sea válido como nombre de colección
        collection_name = f"cv_{name.lower().replace(' ', '_').replace('-', '_')}"
        self.vector_store = Chroma.from_documents(
            docs_chunked, 
            embedding=embeddings,
            collection_name=collection_name
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def retrieve(self, question: str) -> Tuple[str, List[Document]]:
        """Retrieve contextos relevantes para una pregunta."""
        docs = self.retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        return context, docs

    def get_context(self, question: str) -> Tuple[str, List[Document]]:
        """Alias para retrieve."""
        return self.retrieve(question)


# -----------------------------
# Helpers
# -----------------------------


def load_docs_from_pdf(path: str) -> List[Document]:
    """Carga un PDF y lo transforma en documentos LangChain."""
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs


def is_comparative_query(query: str, llm: ChatOpenAI) -> bool:
    """
    Detecta si una pregunta requiere comparación o selección entre múltiples personas.
    
    Args:
        query: Pregunta del usuario
        llm: Modelo LLM para detección
        
    Returns:
        True si la pregunta requiere comparación/selección, False en caso contrario
    """
    prompt_text = f"""Analiza la siguiente pregunta y determina si requiere comparar o seleccionar entre múltiples personas/candidatos.

Pregunta: "{query}"

Responde SOLO con "true" o "false" (sin comillas, sin texto adicional).

Una pregunta es comparativa si:
- Pregunta quién es el mejor/más adecuado para algo
- Pregunta quién tiene más/menos de algo
- Pregunta comparar habilidades/experiencias entre personas
- Pregunta recomendar quién para un trabajo/rol
- Pregunta diferencias entre candidatos

Ejemplos:
- "¿Quién es el mejor fit para programación?" → true
- "¿Quién tiene más experiencia en Python?" → true
- "Compara las habilidades de los candidatos" → true
- "¿Quién recomiendas para este trabajo?" → true
- "¿Qué experiencia tiene Juan?" → false
- "¿Dónde estudia María?" → false
- "¿Qué tecnologías usa?" → false

Respuesta (solo true o false):"""

    try:
        response = llm.invoke(prompt_text)
        content = response.content.strip().lower()
        
        # Limpiar respuesta si tiene markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[-1].strip() == "```":
                content = "\n".join(lines[1:-1]).strip().lower()
            else:
                content = "\n".join(lines[1:]).strip().lower()
        
        return content == "true" or content.startswith("true")
    except Exception as e:
        # Si falla la detección, usar heurística simple
        comparative_keywords = [
            "mejor", "peor", "más", "menos", "comparar", "comparación",
            "recomendar", "recomendación", "fit", "adecuado", "suitable",
            "diferencias", "quién", "cual", "seleccionar", "elegir"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in comparative_keywords)


def detect_people_in_query(
    query: str, available_people: List[str], llm: ChatOpenAI
) -> List[str]:
    """
    Detecta qué personas se mencionan en la query usando un LLM.
    
    Args:
        query: Pregunta del usuario
        available_people: Lista de nombres de personas disponibles
        llm: Modelo LLM para detección
        
    Returns:
        Lista de nombres detectados (puede estar vacía)
    """
    if not available_people:
        return []

    prompt_text = f"""Analiza la siguiente pregunta y determina si menciona alguna de estas personas: {', '.join(available_people)}

Pregunta: "{query}"

Responde SOLO con un JSON array de los nombres mencionados. Si no se menciona ninguna persona, responde con un array vacío [].

Ejemplos:
- "¿Qué experiencia tiene Juan?" → ["Juan"]
- "¿Dónde estudia María?" → ["María"]
- "Compara las habilidades de Juan y Pedro" → ["Juan", "Pedro"]
- "¿Qué tecnologías usa?" → []

Respuesta (solo JSON, sin texto adicional):"""

    try:
        response = llm.invoke(prompt_text)
        content = response.content.strip()

        # Limpiar respuesta si tiene markdown code blocks
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[-1].strip() == "```":
                content = "\n".join(lines[1:-1])
            else:
                content = "\n".join(lines[1:])
        elif content.startswith("```json"):
            lines = content.split("\n")
            if lines[-1].strip() == "```":
                content = "\n".join(lines[1:-1])
            else:
                content = "\n".join(lines[1:])

        detected = json.loads(content)
        # Filtrar solo nombres que están en available_people
        detected = [name for name in detected if name in available_people]
        return detected
    except Exception as e:
        st.sidebar.warning(f"Error en detección de personas: {e}. Usando agente por defecto.")
        return []


def combine_contexts(contexts: Dict[str, Tuple[str, List[Document]]]) -> str:
    """
    Combina contextos de múltiples agentes en un formato estructurado.
    
    Args:
        contexts: Dict con nombre_persona -> (contexto_texto, documentos)
        
    Returns:
        String con contexto combinado y organizado por persona
    """
    combined = []
    for person_name, (context_text, _) in contexts.items():
        combined.append(f"=== CV de {person_name} ===\n{context_text}\n")
    return "\n".join(combined)


def route_query(
    query: str,
    agents: Dict[str, PersonAgent],
    default_agent_name: str,
    detection_llm: ChatOpenAI,
    qa_llm: ChatOpenAI,
    prompt_template: PromptTemplate,
) -> Tuple[str, Dict[str, List[Document]], List[str]]:
    """
    Enruta una query a los agentes apropiados y genera respuesta.
    
    Args:
        query: Pregunta del usuario
        agents: Dict con nombre -> PersonAgent
        default_agent_name: Nombre del agente por defecto (alumno)
        detection_llm: Modelo LLM para detección de personas
        qa_llm: Modelo LLM para generar respuestas
        prompt_template: Template del prompt
        
    Returns:
        Tupla con (respuesta, dict de docs por agente, lista de agentes usados)
    """
    available_people = list(agents.keys())
    
    # Detectar si es una pregunta comparativa que requiere todos los CVs
    is_comparative = is_comparative_query(query, detection_llm)
    
    # Si es comparativa, usar todos los agentes disponibles
    if is_comparative and len(agents) > 1:
        contexts_dict = {}
        docs_dict = {}
        for person_name, agent in agents.items():
            context, docs = agent.retrieve(query)
            contexts_dict[person_name] = (context, docs)
            docs_dict[person_name] = docs
        
        combined_context = combine_contexts(contexts_dict)
        formatted_prompt = prompt_template.format(
            context=combined_context, question=query
        )
        response = qa_llm.invoke(formatted_prompt)
        return response.content, docs_dict, list(agents.keys())
    
    # Detección normal de personas mencionadas
    detected_people = detect_people_in_query(query, available_people, detection_llm)

    # Si no se detecta ninguna persona, usar agente por defecto
    if not detected_people:
        if default_agent_name in agents:
            agent = agents[default_agent_name]
            context, docs = agent.retrieve(query)
            formatted_prompt = prompt_template.format(
                context=f"=== CV de {default_agent_name} ===\n{context}",
                question=query,
            )
            response = qa_llm.invoke(formatted_prompt)
            return (
                response.content,
                {default_agent_name: docs},
                [default_agent_name],
            )
        else:
            return (
                "No hay agentes disponibles. Por favor carga al menos un CV.",
                {},
                [],
            )

    # Si se detecta una persona, usar su agente
    if len(detected_people) == 1:
        person_name = detected_people[0]
        agent = agents[person_name]
        context, docs = agent.retrieve(query)
        formatted_prompt = prompt_template.format(
            context=f"=== CV de {person_name} ===\n{context}",
            question=query,
        )
        response = qa_llm.invoke(formatted_prompt)
        return response.content, {person_name: docs}, [person_name]

    # Si se detectan múltiples personas, combinar contextos
    contexts_dict = {}
    docs_dict = {}
    for person_name in detected_people:
        agent = agents[person_name]
        context, docs = agent.retrieve(query)
        contexts_dict[person_name] = (context, docs)
        docs_dict[person_name] = docs

    combined_context = combine_contexts(contexts_dict)
    formatted_prompt = prompt_template.format(
        context=combined_context, question=query
    )
    response = qa_llm.invoke(formatted_prompt)
    return response.content, docs_dict, detected_people


def create_prompt_template() -> PromptTemplate:
    """Crea el template del prompt para respuestas."""
    template = """Eres un asistente que responde preguntas sobre CVs de integrantes de un equipo.

Debes responder **únicamente** usando la información del contexto proporcionado.
Si la respuesta no está en el contexto, responde exactamente:
"No tengo esa información en el/los CV(s)."

Cuando el contexto incluye múltiples CVs:
- Si la pregunta es comparativa (ej: "¿quién es mejor para...?", "¿quién tiene más...?"), compara explícitamente entre todos los candidatos y proporciona una recomendación clara.
- Organiza tu respuesta claramente indicando de quién es cada información.
- Para preguntas comparativas, estructura tu respuesta comparando punto por punto y concluye con una recomendación.

📄 CONTEXTO:
{context}

❓ PREGUNTA:
{question}

🧠 RESPUESTA clara, en español y bien estructurada:"""
    return PromptTemplate(input_variables=["context", "question"], template=template)


# -----------------------------
# Sidebar: configuración y carga de CVs
# -----------------------------
st.sidebar.header("⚙️ Configuración")

# API Key
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

# LLM para detección (usar modelo rápido)
detection_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=api_key,
)

# LLM para respuestas
qa_llm = ChatOpenAI(
    model=model_name,
    temperature=temperature,
    api_key=api_key,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📄 CVs del Equipo (máximo 3)")

# Inicializar estado de agentes
if "agents" not in st.session_state:
    st.session_state.agents = {}
if "agent_names" not in st.session_state:
    st.session_state.agent_names = []
if "default_agent_name" not in st.session_state:
    st.session_state.default_agent_name = None

# Carga de CVs
MAX_AGENTS = 3
uploaded_files = []
agent_names_input = []

for i in range(MAX_AGENTS):
    st.sidebar.markdown(f"#### Persona {i+1}")
    name = st.sidebar.text_input(
        f"Nombre de la persona {i+1}",
        key=f"name_{i}",
        placeholder="Ej: Lucas Argento",
        help="Nombre de la persona (usado para detección en queries)",
    )
    uploaded_file = st.sidebar.file_uploader(
        f"CV {i+1} (PDF)",
        type=["pdf"],
        key=f"cv_{i}",
        help=f"Sube el CV de {name if name else 'la persona'}",
    )

    if uploaded_file is not None and name:
        uploaded_files.append((name, uploaded_file))
        agent_names_input.append(name)

# Establecer agente por defecto (primera persona cargada = alumno)
if agent_names_input and st.session_state.default_agent_name is None:
    st.session_state.default_agent_name = agent_names_input[0]
    st.sidebar.info(f"✅ Agente por defecto: **{agent_names_input[0]}**")

# Botón para actualizar agentes
if st.sidebar.button("🔄 Cargar/Actualizar Agentes"):
    if not uploaded_files:
        st.sidebar.error("Por favor carga al menos un CV con su nombre.")
    else:
        with st.spinner("Creando agentes..."):
            new_agents = {}
            new_agent_names = []

            for name, uploaded_file in uploaded_files:
                # Guardar PDF temporalmente con nombre único
                # Resetear el puntero del archivo para asegurar que leemos desde el inicio
                uploaded_file.seek(0)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf", prefix=f"cv_{name}_") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    # Cargar documentos del PDF
                    docs = load_docs_from_pdf(tmp_path)
                    
                    # Verificar que se cargaron documentos
                    if not docs:
                        st.sidebar.warning(f"⚠️ El CV de {name} está vacío o no se pudo leer.")
                        continue
                    
                    # Crear agente con documentos únicos
                    agent = PersonAgent(name, docs, model_name, temperature, api_key)
                    new_agents[name] = agent
                    new_agent_names.append(name)
                    st.sidebar.success(f"✅ Agente creado para **{name}** ({len(docs)} páginas)")
                    
                    # Limpiar archivo temporal después de cargar
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass  # Ignorar errores al eliminar archivo temporal
                        
                except Exception as e:
                    st.sidebar.error(f"❌ Error creando agente para {name}: {e}")
                    # Intentar limpiar archivo temporal en caso de error
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            if new_agents:
                st.session_state.agents = new_agents
                st.session_state.agent_names = new_agent_names
                if not st.session_state.default_agent_name:
                    st.session_state.default_agent_name = new_agent_names[0]
                st.sidebar.success(f"✅ {len(new_agents)} agente(s) listo(s)")

# Mostrar agentes activos
st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Agentes Activos")
if st.session_state.agents:
    for name in st.session_state.agent_names:
        default_badge = " (Por defecto)" if name == st.session_state.default_agent_name else ""
        st.sidebar.markdown(f"- **{name}**{default_badge}")
else:
    st.sidebar.info("No hay agentes cargados. Sube CVs y haz clic en 'Cargar/Actualizar Agentes'.")

# -----------------------------
# Verificar que hay agentes cargados
# -----------------------------
if not st.session_state.agents:
    st.info(
        "👋 **Bienvenido al Sistema Multi-Agente para CVs**\n\n"
        "1. En la barra lateral, ingresa el nombre de cada integrante del equipo\n"
        "2. Sube su CV en PDF\n"
        "3. Haz clic en 'Cargar/Actualizar Agentes'\n"
        "4. ¡Empieza a hacer preguntas! El sistema detectará automáticamente sobre quién preguntas.\n\n"
        "**Ejemplos de preguntas:**\n"
        "- '¿Qué experiencia tiene [Nombre]?' (pregunta sobre una persona)\n"
        "- 'Compara las habilidades de [Nombre1] y [Nombre2]' (pregunta sobre múltiples personas)\n"
        "- '¿Qué tecnologías usa?' (sin mencionar nombre, usa agente por defecto)"
    )
    st.stop()

# Crear prompt template
prompt_template = create_prompt_template()

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
# Input tipo chat
# -----------------------------
query = st.chat_input("Pregunta sobre los CVs del equipo...")

if query:
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(query)

    # Ejecutar routing y RAG
    with st.spinner("🤔 Procesando consulta..."):
        answer, docs_dict, active_agents = route_query(
            query,
            st.session_state.agents,
            st.session_state.default_agent_name,
            detection_llm,
            qa_llm,
            prompt_template,
        )

    # Mostrar agentes activos
    if active_agents:
        agent_badges = " | ".join([f"🤖 **{name}**" for name in active_agents])
        st.info(f"**Agentes activos:** {agent_badges}")

    # Mostrar respuesta del asistente
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Guardar en historial
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Mostrar contexto recuperado por agente
    if docs_dict:
        with st.expander("🔍 Ver fragmentos de CVs usados para responder"):
            for person_name, docs in docs_dict.items():
                st.markdown(f"### 📄 CV de **{person_name}**")
                for i, d in enumerate(docs, start=1):
                    st.markdown(f"**Fragmento {i}:**")
                    st.write(
                        d.page_content[:600]
                        + ("..." if len(d.page_content) > 600 else "")
                    )
                    st.markdown("---")

# Mensaje inicial si todavía no hay conversación
if not st.session_state.history and query is None:
    st.info(
        "💡 **Ejemplos de preguntas:**\n\n"
        f"- '¿Qué experiencia tiene {st.session_state.agent_names[0] if st.session_state.agent_names else "[Nombre]"}?'\n"
        "- '¿Dónde estudia [Nombre]?'\n"
        "- 'Compara las habilidades técnicas de [Nombre1] y [Nombre2]'\n"
        "- '¿Qué tecnologías usa?' (sin mencionar nombre, usa agente por defecto)"
    )

