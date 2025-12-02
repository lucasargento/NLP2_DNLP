"""
Chatbot con Memoria Persistente usando LangChain y Groq
=====================================================

Este archivo implementa un chatbot con interfaz web usando Streamlit que:
- Mantiene memoria de conversaciones anteriores
- Utiliza diferentes modelos de LLM a través de Groq
- Permite personalización del comportamiento del bot
- Gestiona la memoria conversacional automáticamente

Tecnologías utilizadas:
- Streamlit: Para la interfaz web
- LangChain: Para gestión de memoria y cadenas de conversación
- Groq: Como proveedor de modelos LLM
- Python: Lenguaje de programación

Autor: Clase VI - CEIA LLMIAG
Curso: Large Language Models y Generative AI

Instrucciones para ejecutar:
    streamlit run chatbot_gestionada.py

Requisitos:
    pip install streamlit groq langchain langchain-groq

Variables de entorno necesarias:
    GROQ_API_KEY: Tu clave API de Groq (obtener en https://console.groq.com)
"""

# ========================================
# IMPORTACIÓN DE LIBRERÍAS NECESARIAS
# ========================================

import streamlit as st           # Framework para crear aplicaciones web interactivas
from groq import Groq           # Cliente oficial de Groq para acceso a LLMs
import random                   # Para funcionalidades aleatorias (si se necesitan)
import os                      # Para acceso a variables de entorno

# Importaciones específicas de LangChain para gestión de conversaciones

from langchain_core.prompts import (
    ChatPromptTemplate,           # Template para estructurar mensajes de chat
    HumanMessagePromptTemplate,   # Template específico para mensajes humanos
    MessagesPlaceholder,          # Marcador de posición para el historial
)
from langchain_core.messages import SystemMessage  # Mensajes del sistema
from langchain_groq import ChatGroq              # Integración LangChain-Groq
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

def main():
    """
    Función principal de la aplicación de chatbot.
    
    Esta función coordina todos los componentes del chatbot:
    1. Configuración de la interfaz de usuario
    2. Gestión de la memoria conversacional
    3. Integración con el modelo de lenguaje
    4. Procesamiento de preguntas y respuestas
    
    Funcionalidades principales:
    - Interfaz web responsiva con Streamlit
    - Memoria de conversación con longitud configurable
    - Selección de diferentes modelos LLM
    - Personalización del prompt del sistema
    - Historial persistente durante la sesión
    """
    
    # ========================================
    # CONFIGURACIÓN INICIAL Y AUTENTICACIÓN
    # ========================================
    
    # Obtener la clave API de Groq desde las variables de entorno
    # Esto es una práctica de seguridad recomendada para no exponer credenciales en el código
    groq_api_key = os.getenv('GROQ_API_KEY')
    
    # Verificar si la clave API está configurada
    if not groq_api_key:
        st.error("⚠️ GROQ_API_KEY no está configurada en las variables de entorno")
        st.info("💡 Configura tu clave API: export GROQ_API_KEY='tu-clave-aqui'")
        st.stop()  # Detener la ejecución si no hay clave API

    # ========================================
    # CONFIGURACIÓN DE LA INTERFAZ PRINCIPAL
    # ========================================
    
    # Configurar el título y descripción de la aplicación
    st.title("🤖 Chatbot CEIA con Memoria Persistente")
    st.markdown("""
    **¡Bienvenido al chatbot educativo!** 
    
    Este chatbot utiliza:
    - 🧠 **Memoria conversacional**: Recuerda el contexto de tu conversación
    - 🔄 **Modelos intercambiables**: Puedes elegir diferentes LLMs
    - ⚙️ **Personalización**: Configura el comportamiento del asistente
    - 🚀 **Powered by Groq**: Respuestas rápidas y precisas
    """)

    # ========================================
    # PANEL DE CONFIGURACIÓN LATERAL
    # ========================================
    
    st.sidebar.title('⚙️ Configuración del Chatbot')
    st.sidebar.markdown("---")
    
    # Input para el prompt del sistema - Define la personalidad y comportamiento del bot
    st.sidebar.subheader("🎭 Personalidad del Bot")
    system_prompt = st.sidebar.text_area(
        "Mensaje del sistema:",
        value="Eres un asistente educativo especializado en inteligencia artificial y machine learning. Responde de manera clara, didáctica y con ejemplos prácticos.",
        height=100,
        help="Define cómo debe comportarse el chatbot. Ejemplo: 'Eres un profesor de matemáticas que explica conceptos de forma simple'"
    )
    
    # Selector de modelo LLM disponible en Groq
    st.sidebar.subheader("🧠 Modelo de Lenguaje")
    model = st.sidebar.selectbox(
        'Elige un modelo:',
        [
            'llama-3.1-8b-instant',   # Reemplazo recomendado para 8B
            'llama-3.3-70b-versatile' # Reemplazo recomendado para 70B
        ],
        help="Diferentes modelos tienen distintas capacidades y velocidades"
    )
    
    # Información sobre el modelo seleccionado
    model_info = {
        'llama-3.1-8b-instant': "🦙 Llama 3.1 8B Instant: excelente precio-rendimiento y baja latencia",
        'llama-3.3-70b-versatile': "🦙 Llama 3.3 70B Versatile: mayor calidad general"
    }
    st.sidebar.info(model_info.get(model, "Modelo seleccionado"))
    
    # Control deslizante para la longitud de memoria
    st.sidebar.subheader("🧠 Configuración de Memoria")
    conversational_memory_length = st.sidebar.slider(
        'Longitud de la memoria conversacional:', 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Número de intercambios anteriores que el bot recordará. Más memoria = mayor contexto pero mayor costo computacional"
    )
    
    # Mostrar información sobre la memoria
    st.sidebar.caption(f"💭 El bot recordará los últimos {conversational_memory_length} intercambios")

    # ========================================
    # CONFIGURACIÓN DE LA MEMORIA CONVERSACIONAL
    # ========================================
    
    # Nueva API: gestionamos historial con RunnableWithMessageHistory + InMemoryChatMessageHistory
    if "session_id" not in st.session_state:
        st.session_state.session_id = "default"
    if "history_store" not in st.session_state:
        st.session_state.history_store = {}
    
    # ========================================
    # GESTIÓN DEL HISTORIAL DE CONVERSACIÓN
    # ========================================
    
    # Inicializar el historial de chat en el estado de la sesión de Streamlit
    # st.session_state permite mantener datos entre ejecuciones de la aplicación
    if 'historial_chat' not in st.session_state:
        st.session_state.historial_chat = []
        st.sidebar.success("💬 Nueva conversación iniciada")
    else:
        # Mostrar información del historial en la barra lateral
        st.sidebar.info(f"💬 Conversación con {len(st.session_state.historial_chat)} mensajes")
    
    # Botón para limpiar el historial
    if st.sidebar.button("🗑️ Limpiar Conversación"):
        st.session_state.historial_chat = []
        # Reiniciar historial de LangChain para la sesión actual
        sid = st.session_state.session_id
        if sid in st.session_state.history_store:
            st.session_state.history_store[sid] = InMemoryChatMessageHistory()
        st.sidebar.success("✅ Conversación limpiada")
        st.rerun()  # Recargar la aplicación
    
    # ========================================
    # INTERFAZ DE ENTRADA DEL USUARIO
    # ========================================
    
    # Crear el campo de entrada para las preguntas del usuario
    st.markdown("### 💬 Haz tu pregunta:")
    user_question = st.text_input(
        "Escribe tu mensaje aquí:",
        placeholder="Por ejemplo: ¿Qué es el machine learning?",
        label_visibility="collapsed"
    )


    # ========================================
    # CONFIGURACIÓN DEL MODELO DE LENGUAJE
    # ========================================
    
    # Inicializar el cliente de ChatGroq con las configuraciones seleccionadas
    try:
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,     # Clave API para autenticación
            model_name=model,              # Modelo seleccionado por el usuario
            temperature=0.7,               # Creatividad de las respuestas (0=determinista, 1=creativo)
            max_tokens=1000,               # Máximo número de tokens en la respuesta
        )
        st.sidebar.success("✅ Modelo conectado correctamente")
    except Exception as e:
        st.sidebar.error(f"❌ Error al conectar con Groq: {str(e)}")
        st.stop()

    # ========================================
    # PROCESAMIENTO DE LA PREGUNTA Y RESPUESTA
    # ========================================

    # Si el usuario ha hecho una pregunta,
    if user_question and user_question.strip():

        # Mostrar indicador de carga mientras se procesa
        with st.spinner('🤔 El chatbot está pensando...'):
            
            try:
                # ========================================
                # CONSTRUCCIÓN DEL TEMPLATE DE CONVERSACIÓN
                # ========================================
                
                # Crear un template de chat que incluye:
                # 1. Mensaje del sistema (personalidad/instrucciones)
                # 2. Historial de conversación (memoria)
                # 3. Mensaje actual del usuario
                prompt = ChatPromptTemplate.from_messages([
                    
                    # Mensaje del sistema - Define el comportamiento del chatbot
                    SystemMessage(content=system_prompt),
                    
                    # Marcador de posición para el historial - Se reemplaza automáticamente
                    MessagesPlaceholder(variable_name="historial_chat"),
                    
                    # Template para el mensaje actual del usuario
                    HumanMessagePromptTemplate.from_template("{human_input}")
                ])
                
                # ========================================
                # CREACIÓN DE LA CADENA DE CONVERSACIÓN
                # ========================================
                
                # Nueva composición: Prompt → Modelo
                chain = prompt | groq_chat
                
                # Obtener/crear historial de la sesión actual
                session_id = st.session_state.session_id
                store = st.session_state.history_store
                if session_id not in store:
                    store[session_id] = InMemoryChatMessageHistory()
                
                # Envolver con memoria conversacional usando la nueva API
                chain_with_memory = RunnableWithMessageHistory(
                    chain,
                    lambda sid: store.setdefault(sid, InMemoryChatMessageHistory()),
                    input_messages_key="human_input",
                    history_messages_key="historial_chat",
                )
                
                # ========================================
                # GENERACIÓN DE LA RESPUESTA
                # ========================================
                
                # Enviar la pregunta al modelo y obtener la respuesta
                result = chain_with_memory.invoke(
                    {"human_input": user_question},
                    config={"configurable": {"session_id": session_id}},
                )
                response = getattr(result, "content", result)
                
                # ========================================
                # ALMACENAMIENTO Y VISUALIZACIÓN
                # ========================================
                
                # Crear un objeto mensaje para almacenar en el historial
                message = {'humano': user_question, 'IA': response}
                
                # Agregar el mensaje al historial de la sesión
                st.session_state.historial_chat.append(message)
                
                # ========================================
                # MOSTRAR LA CONVERSACIÓN
                # ========================================
                
                # Mostrar la respuesta actual destacada
                st.markdown("### 🤖 Respuesta:")
                st.markdown(f"""
                <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                    {response}
                </div>
                """, unsafe_allow_html=True)
                
                # Información adicional sobre la respuesta
                st.caption(f"📊 Modelo: {model} | 🧠 Memoria: {conversational_memory_length} mensajes")
                
            except Exception as e:
                # Manejo de errores durante el procesamiento
                st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                st.info("💡 Verifica tu conexión a internet y la configuración de la API")

    # ========================================
    # INFORMACIÓN ADICIONAL PARA ESTUDIANTES
    # ========================================
    
    # Panel expandible con información educativa
    with st.expander("📚 Información Técnica para Estudiantes"):
        st.markdown("""
        **¿Cómo funciona este chatbot?**
        
        1. **Memoria Conversacional**: Utiliza `ConversationBufferWindowMemory` para recordar contexto
        2. **Templates de Prompts**: Estructura los mensajes de manera consistente
        3. **Cadenas LLM**: `LLMChain` conecta el modelo con la lógica de conversación
        4. **Estado de Sesión**: Streamlit mantiene el historial durante la sesión
        5. **Integración Groq**: Acceso rápido a modelos de lenguaje modernos
        
        **Conceptos Clave:**
        - **System Prompt**: Define la personalidad del chatbot
        - **Memory Window**: Controla cuánto contexto previo se incluye
        - **Token Limits**: Gestiona el costo y velocidad de las respuestas
        - **Model Selection**: Diferentes modelos para diferentes necesidades
        
        **Arquitectura del Sistema:**
        ```
        Usuario → Streamlit → LangChain → Groq → LLM → Respuesta
                     ↓
               Session State (Memoria)
        ```
        """)
    
    # Pie de página con información del curso
    st.markdown("---")
    st.markdown("**📖 Clase VI - CEIA LLMIAG** | Ejemplo educativo de chatbot con memoria persistente")


if __name__ == "__main__":
    # Punto de entrada de la aplicación
    # Solo ejecutar main() si este archivo se ejecuta directamente
    main()
