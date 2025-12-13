# TP3: Sistema Multi-Agente para Consulta de CVs

## 📋 Descripción

Sistema de agentes RAG (Retrieval-Augmented Generation) que permite consultar múltiples CVs de integrantes de un equipo. Cada integrante tiene su propio agente especializado que responde preguntas sobre su CV.

### Características Principales

- ✅ **Soporte Multi-Agente**: Hasta 3 agentes (uno por integrante del equipo)
- ✅ **Detección Inteligente**: Identifica automáticamente qué persona(s) se mencionan en la query
- ✅ **Agente por Defecto**: Cuando no se menciona ninguna persona, usa el agente del alumno
- ✅ **Consultas Multi-Persona**: Combina contextos de múltiples CVs cuando se consultan varias personas
- ✅ **Interfaz Intuitiva**: UI clara que muestra qué agente(s) están procesando cada consulta

## 🏗️ Arquitectura

El sistema está compuesto por:

1. **PersonAgent**: Clase que encapsula un agente RAG individual
   - Vector store (Chroma) con embeddings del CV
   - Retriever configurado para búsqueda semántica
   - Métodos para recuperar contextos relevantes

2. **Sistema de Detección**: Usa un LLM para identificar nombres de personas en las queries
   - Analiza la pregunta del usuario
   - Extrae nombres mencionados
   - Mapea a agentes disponibles

3. **Router Multi-Agente**: Enruta queries a los agentes apropiados
   - 0 personas mencionadas → Agente por defecto (alumno)
   - 1 persona mencionada → Agente específico
   - Múltiples personas → Combina contextos de todos los agentes relevantes

4. **Combinación de Contextos**: Fusiona información de múltiples CVs cuando es necesario

## 🚀 Instalación y Uso

### Requisitos

- Python 3.10 - 3.12
- OpenAI API Key
- CVs en formato PDF (hasta 3)

### Pasos

1. **Clonar el repositorio** (si aplica)

2. **Crear entorno virtual**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**:
   ```bash
   # Asegúrate de estar en el directorio TP3
   cd TPs/TP3
   
   # Activa el entorno virtual
   source .venv/bin/activate  # En Windows: .venv\Scripts\activate
   
   # Ejecuta la aplicación
   streamlit run app.py
   ```

   **Nota**: Si encuentras un error sobre directorio temporal (`FileNotFoundError: No usable temporary directory`), el código ya incluye una solución automática. Si persiste, puedes configurar manualmente:
   ```bash
   export TMPDIR=/tmp
   streamlit run app.py
   ```

5. **Configurar en la UI**:
   - Ingresar OpenAI API Key en la barra lateral
   - Para cada integrante (hasta 3):
     - Ingresar su nombre
     - Subir su CV en PDF
   - Hacer clic en "🔄 Cargar/Actualizar Agentes"
   - ¡Empezar a hacer preguntas!

## 💡 Ejemplos de Uso

### Pregunta sobre una persona específica
```
¿Qué experiencia tiene Lucas Argento?
```
→ El sistema detecta "Lucas Argento" y enruta al agente correspondiente.

### Pregunta sin mencionar nombre
```
¿Qué tecnologías usa?
```
→ El sistema usa el agente por defecto (primera persona cargada).

### Pregunta comparando múltiples personas
```
Compara las habilidades técnicas de Juan y María
```
→ El sistema combina contextos de ambos CVs y genera una respuesta comparativa.

### Pregunta sobre experiencia de múltiples personas
```
¿Dónde trabajaron Pedro y Ana?
```
→ El sistema recupera información de ambos CVs y responde de manera organizada.

## 🔧 Configuración

### Modelos Disponibles
- `gpt-4o-mini` (recomendado, rápido y económico)
- `gpt-4o` (más preciso, más costoso)
- `gpt-3.5-turbo` (alternativa económica)

### Parámetros Ajustables
- **Temperature**: Controla la creatividad de las respuestas (0.0 = determinista, 1.0 = creativo)
- **Chunk Size**: Tamaño de los fragmentos del CV (700 caracteres por defecto)
- **Top K**: Número de fragmentos recuperados por agente (3 por defecto)

## 📊 Flujo de Ejecución

1. **Inicialización**: Usuario carga CVs con nombres asociados
2. **Creación de Agentes**: Se crea un `PersonAgent` por cada CV cargado
3. **Procesamiento de Query**:
   - Usuario envía pregunta
   - Sistema detecta personas mencionadas (si las hay)
   - Router selecciona agente(s) apropiado(s)
   - Cada agente relevante ejecuta retrieve
   - Se combinan contextos si hay múltiples agentes
   - Se genera respuesta final
4. **Visualización**: 
   - Respuesta del asistente
   - Indicador de agentes activos
   - Fragmentos de CVs usados (expandible)

## 🎯 Funcionalidades del Video Demo

Para la demostración en video, asegúrate de mostrar:

1. ✅ Carga de múltiples CVs (hasta 3) con nombres
2. ✅ Query sin mencionar nombre → usa agente por defecto
3. ✅ Query mencionando una persona específica → usa su agente
4. ✅ Query mencionando múltiples personas → combina contextos
5. ✅ Visualización de fragmentos de cada CV usado
6. ✅ Indicadores de agentes activos

## 📝 Notas Técnicas

- **Detección de Nombres**: Usa `gpt-4o-mini` con prompt estructurado para extraer nombres de manera eficiente
- **Persistencia**: Los vector stores se mantienen en memoria durante la sesión de Streamlit
- **Performance**: Los agentes se crean solo cuando se suben CVs nuevos o se actualiza la configuración
- **Fallback**: Si la detección de nombres falla, el sistema usa el agente por defecto automáticamente

## 🛠️ Tecnologías Utilizadas

- **Streamlit**: Interfaz web interactiva
- **LangChain**: Framework para aplicaciones con LLMs
- **Chroma**: Vector database para almacenamiento de embeddings
- **OpenAI**: Embeddings (text-embedding-3-small) y LLMs (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- **PyPDF**: Carga y procesamiento de PDFs

## 📚 Estructura del Proyecto

```
TP3/
├── app.py              # Aplicación principal
├── requirements.txt     # Dependencias
└── README.md          # Esta documentación
```

## 👤 Autor

Lucas Argento - CEIA LLMIAG - Diplomatura en Inteligencia Artificial

## 📄 Licencia

Este trabajo es parte del material académico de la Diplomatura en IA del CEIA - FIUBA.

