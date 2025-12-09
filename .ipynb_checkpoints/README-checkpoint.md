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

---

## ▶️ Cómo ejecutar el TP2 (Chatbot RAG)

### 🔧 Requisitos

- Python **3.10 — 3.12**
- **OpenAI API Key**
- Tu CV en PDF (o usar el default incluido)

### 🚀 Pasos

1. **Clonar el repo**
2. Generar un venv:
3. instalar requirements.txt
4. streamlit run app.py