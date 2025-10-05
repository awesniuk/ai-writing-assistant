# ✍️ AI Writing Assistant

A privacy-conscious, author-style-preserving AI writing assistant built using **LLaMA 2**, **LoRA fine-tuning**, **FastAPI**, **Qdrant**, and **Retrieval-Augmented Generation (RAG)**. The system can emulate the writing style of a specific author and generate context-aware outputs grounded in a local semantic knowledge base.

---

# 🧠 Overview

This application combines a fine-tuned LLaMA 2 model with a RAG pipeline to produce text aligned with a target authors writing style. Fine-tuning is performed using **LoRA (Low-Rank Adaptation)**, and the assistant supports semantic search over a local vector store to inject contextual information into the generation process.

---

## ⚙️ Architecture

```
flowchart TD
    A[User Input] --> B[Embed Input Query]
    B --> C[Semantic Search (Qdrant)]
    C --> D[Retrieve Relevant Chunks]
    D --> E[Augment Prompt]
    E --> F[LLM Generation (LLaMA 2 + LoRA)]
    F --> G[Styled Response]
```

---

## 🚀 Tech Stack

| Component           | Description                                                    |
|---------------------|----------------------------------------------------------------|
| **LLaMA 2 7B Chat** | Base LLM used for author-style generation (meta-llama/Llama-2-7b-chat-hf) |
| **LoRA (PEFT)**     | Lightweight fine-tuning for efficient personalization          |
| **SentenceTransformers** | Embedding model for semantic vector search (all-MiniLM-L6-v2)  |
| **Qdrant**          | Vector DB for fast retrieval of relevant document chunks       |
| **FastAPI**         | Backend REST API serving the generation and retrieval endpoints |
| **SQLite**          | Lightweight, local relational DB for storing persistent data    |
| **Uvicorn**         | ASGI server used to run the FastAPI app                         |
| **Frontend**        | Placeholder for UI components (to be implemented)               |

---

## 📂 Project Structure

```
ai-writing-assistant/
├── backend/  
│   ├── db.py               # Database connection and logic  
│   ├── main.py             # FastAPI entry point  
│   ├── model.py            # LLM model loading and inference  
│   ├── rag.py              # RAG pipeline logic (retrieval + augmentation)  
│   └── schemas.py          # Pydantic models for API I/O  
├── data/  
│   └── database.db         # SQLite database  
├── fine_tuning/  
│   ├── config.py           # Training configuration  
│   ├── data/  
│   │   └── author_snippets.json     # Raw snippets used for fine-tuning  
│   ├── dataset.jsonl       # Preprocessed training dataset  
│   ├── lora_output/        # LoRA fine-tuned model artifacts  
│   ├── test_training_args.py   # Unit test for training config  
│   └── train_lora.py       # Script to fine-tune LLaMA using PEFT + LoRA  
├── frontend/               # Frontend code (to be implemented)  
├── requirements.txt        # Python dependencies  
└── README.md               # Project documentation
```
---

## 🔧 Local Setup

**Prerequisites:** Python 3.8+, git, virtualenv.

> **Note:** You must request access to LLaMA 2 from Meta and download it via Hugging Face.

```bash
# 1. Clone the repository
git clone https://github.com/awesniuk/ai-writing-assistant.git
cd ai-writing-assistant

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Install additional dependencies for training/inference
pip install peft bitsandbytes accelerate transformers datasets
pip install --upgrade transformers peft

# 5. Configure Accelerate (for multi-GPU, 4-bit, or performance tuning)
accelerate config

# 6. Run the FastAPI server
uvicorn backend.main:app --reload
```
---

## 🧪 Example Use Cases

- ✍️ Personalized ghostwriting in the style of an author or brand  
- 📚 AI editorial assistants for authors or journalists  
- 🏷️ Consistent brand voice across blogs, emails, and ads  
- 🧠 Knowledge-grounded assistant for custom corpora  

---

## 🔬 Fine-Tuning Notes

Uses LoRA (via Hugging Face PEFT) for efficient fine-tuning on small author-specific datasets.

Training data should be formatted as JSONL with the structure:

```json
{"instruction": "...", "input": "...", "output": "..."}

Fine-tuned model weights are saved in /fine_tuning/lora_output/.
```
---

## 📄 License

This project is released under the MIT License.
Use of the LLaMA 2 model is subject to Meta’s LLaMA license.
