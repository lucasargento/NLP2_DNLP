# CEIA-LLMIAG — NLP2_DNLP

Este repositorio contiene mi trabajo personal para la materia **NLP II (Deep NLP)** de la Diplomatura en Inteligencia Artificial del **CEIA - FIUBA**.  
Las clases, materiales conceptuales y contenidos académicos pertenecen a los docentes de la materia.  
Todo el código, experimentos, mejoras e implementaciones dentro de este repo fueron realizados por mí.

---

## 📚 Contenidos de la Materia

1. Repaso de Transformers, arquitectura y tokenizers  
2. Arquitecturas de LLMs — Transformer Decoder  
3. Ecosistema actual — APIs, costos, HuggingFace y OpenAI. Evaluación de LLMs  
4. MoEs y técnicas de prompting  
5. Modelos locales y uso de APIs  
6. RAG — Vector DBs, chatbots y práctica  
7. Agentes, fine-tuning y práctica  
8. LLMs de razonamiento — Optimización, generación multimodal y práctica  

---

### 👨‍🏫 Docentes

- **Esp. Abraham Rodriguez** — *abraham.rodz17@gmail.com*  
- **Esp. Ezequiel Guinsburg** — *ezequielguinsburg@gmail.com*

> Nota: este repositorio **no reemplaza** el material oficial. Solo contiene mis desarrollos realizados durante la cursada.

---

## 🧪 Trabajos Prácticos (carpeta `TPS/`)

Todos los trabajos prácticos se encuentran dentro de:

TPS/

| TP | Descripción |
|----|-------------|
| **TP1 — TinyGPT con MoE** | Implementación simplificada estilo GPT con *Mixture of Experts* y pruebas correspondientes. |
| **TP2 — RAG: Chatbot sobre tu CV** | Chatbot con Retrieval-Augmented Generation usando embeddings, VectorDB (Chroma) y Streamlit para interactuar con tu CV. |
| **TP3 — Sistema Multi-Agente para CVs** | Sistema de agentes RAG que permite consultar múltiples CVs (hasta 3). Cada integrante tiene su propio agente especializado con detección inteligente de personas y soporte para consultas comparativas. |

---

## ▶️ Cómo ejecutar los Trabajos Prácticos

### TP2 — Chatbot RAG sobre tu CV

#### 🔧 Requisitos
- Python **3.10 — 3.12**
- **OpenAI API Key**
- Tu CV en PDF (o usar el default incluido)

#### 🚀 Pasos
1. Navegar a la carpeta: `cd TPs/TP2`
2. Crear entorno virtual: `python -m venv .venv`
3. Activar entorno: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
4. Instalar dependencias: `pip install -r requirements.txt`
5. Ejecutar: `streamlit run app.py`

---

### TP3 — Sistema Multi-Agente para Consulta de CVs

#### 🔧 Requisitos
- Python **3.10 — 3.12**
- **OpenAI API Key**
- CVs en formato PDF (hasta 3 integrantes del equipo)

#### 🚀 Pasos
1. Navegar a la carpeta: `cd TPs/TP3`
2. Crear entorno virtual: `python -m venv .venv`
3. Activar entorno: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
4. Instalar dependencias: `pip install -r requirements.txt`
5. Ejecutar: `streamlit run app.py`

#### ✨ Características del TP3
- **Multi-Agente**: Hasta 3 agentes, uno por integrante del equipo
- **Detección Inteligente**: Identifica automáticamente qué persona(s) se mencionan en las queries
- **Consultas Comparativas**: Para preguntas como "¿quién es el mejor fit para...?", usa automáticamente todos los CVs disponibles
- **Agente por Defecto**: Si no se menciona ninguna persona, usa el agente del alumno
- **Combinación de Contextos**: Fusiona información de múltiples CVs cuando se consultan varias personas

#### 💡 Ejemplos de Uso
- `"¿Qué experiencia tiene Lucas?"` → Usa solo el agente de Lucas
- `"¿Quién es el mejor fit para programación?"` → Compara automáticamente todos los CVs
- `"Compara las habilidades de Juan y María"` → Combina contextos de ambos CVs
- `"¿Qué tecnologías usa?"` → Usa agente por defecto (sin mencionar nombre)