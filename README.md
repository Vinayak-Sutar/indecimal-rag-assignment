# Construction Marketplace AI Assistant (Mini-RAG)

*Note: This repository was created as a submission for the Indecimal RAG Assignment.*

This repository contains a Retrieval-Augmented Generation (RAG) pipeline built for **INDECIMAL**, a construction marketplace. It acts as an internal knowledge base assistant, allowing users to ask questions about company policies and specifications. To ensure reliability, the pipeline restricts answers exclusively to the provided internal documentation, strongly preventing AI hallucinations.

## 1. Setup & Local Installation

### Prerequisites

- **Python 3.10+** (Tested on Python 3.10)
- **[Ollama](https://ollama.com/)** (Optional, only required if you want to run offline local inference)
- An **Nvidia GPU** is highly recommended for running local models, though it is not strictly required.

### Installation Instructions

1. Clone the repository and navigate to the project directory.
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Optional: If you intend to run the local Offline LLM, install Ollama and pull the mistral model:
   ```bash
   ollama pull mistral
   ```

### Connecting to Cloud OpenRouter

To use the high-speed cloud fallback, create a `.env` file in the root directory and add your OpenRouter API key:

```env
OPENROUTER_API_KEY=sk-or-your-key-here
```

### Running the System

Launch the interactive Streamlit user interface natively by running:

```bash
streamlit run app.py
```

This will start a local server at `http://localhost:8501`.

---

## 2. Architecture & Design Decisions

### Document Chunking & Search

- **Chunking Strategy:** Used LangChain's `RecursiveCharacterTextSplitter`. The text is split into chunks of `1000` characters with a `200` character overlap to maintain context between paragraphs.
- **Embeddings Model:** HuggingFace's open-source `sentence-transformers/all-MiniLM-L6-v2`. It runs 100% locally on the CPU, offering a great balance between low latency and accurate vector representation without paid APIs.
- **Vector Store & Retrieval:** Selected **FAISS** (Facebook AI Similarity Search) as the localized, in-memory vector database instead of cloud alternatives like Pinecone. The retriever utilizes a dynamic `top_k` cosine-similarity search.

### LLM Integration & Strict Grounding

- **Prompt Engineering (Anti-Hallucination):** The core intelligence relies on LangChain Expression Language (LCEL). To rigidly bind the generation solely to the retrieved context, implemented an exclusionary system prompt:
  > _"If the answer is not in the provided Context, say EXACTLY: 'I cannot answer this question based on the provided documents.'"_
- **Transparency:** Within the Streamlit User Interface, I added an expandable "Retrieved Documents" section. This acts as an explainability anchor—every time the AI replies, it prints out the precise raw chunks it parsed from FAISS.

---

## 3. Evaluation & Quality Analysis

To thoroughly evaluate the pipeline according to standard RAG capabilities, I created a suite of 10 test questions derived from the provided documents and ran them through an automated script (`run_evaluation.py`). The script benchmarks a **Local Model** (Ollama Mistral 7B running on an RTX 3060) against a **Cloud OpenRouter-based LLM** (Nvidia Nemotron 30B).

> **To reproduce the evaluation:** Run `python run_evaluation.py`. The results are saved to `eval_results.json`.

### 1. Comparison: Local vs. OpenRouter Model

- **Answer Quality:** The OpenRouter model (30B) consistently formatted outputs with deeper clarity, using markdown and isolated bullet points. The Local model (7B) yielded functionally accurate answers but with slightly simpler, more concise structures.
- **Latency:** Local inference running natively on the RTX 3060 averaged **~1.5 - 2.8 seconds** per response. The OpenRouter API was faster, averaging **~1.0 - 1.5 seconds** per response.
- **Groundedness to Retrieved Context:** Both models successfully anchored their responses to the RAG context. By maintaining a strict fallback prompt and a low temperature (`0.3`), both models consistently refused to construct answers if the text was missing from the supplied index.

### 2. Quality Analysis

- **Relevance of retrieved chunks:** The local HuggingFace embeddings (`all-MiniLM-L6-v2`) inside the FAISS vector store proved highly relevant. For every document-grounded question (e.g., about GST, Escrow), the nearest-neighbor search consistently retrieved the precise chunk containing the factual answer within the `top_k=3` results.
- **Presence of hallucinations or unsupported claims:** Evaluated via out-of-domain questions ("Who built the Eiffel Tower?", "Who is the CEO?") and missing-information questions ("3 tiers of protection"). Hallucinations were successfully reduced to **0%**. In all test cases where context was absent, the pipeline correctly enforced the hardcoded fallback message: _"I cannot answer this question... "_
- **Completeness and clarity of generated answers:** For in-domain questions, the generated answers completely and accurately resolved the queries without omitting critical definitions (such as correctly returning the "445+ checkpoints" statistic from the quality manual).

### 3. Core 10-Question Evaluation Matrix

| Test Question                                                            | Expected Outcome                     | System Result (Pass/Fail)                                  |
| ------------------------------------------------------------------------ | ------------------------------------ | ---------------------------------------------------------- |
| 1. _What is Indecimal's one-line summary?_                               | Pull 1-line summary                  | **Pass**                                                   |
| 2. _Are the package pricing rates inclusive of GST?_                     | Direct yes/no extraction             | **Pass**                                                   |
| 3. _What is the Escrow-Based Payment Model?_                             | Summarize definitions                | **Pass**                                                   |
| 4. _What kind of home construction does Indecimal support?_              | Aggregate capabilities               | **Pass**                                                   |
| **5. Who built the Eiffel Tower?**                                       | **Guardrail (Force zero-knowledge)** | **Pass** (Refused to answer)                               |
| **6. What are the 3 tiers of protection policies?**                      | **Guardrail (Missing Info Test)**    | **Pass** (Both models refused with temperature=0.3)                    |
| 7. _How does the system ensure quality?_                                 | Map bullet points                    | **Pass**                                                   |
| **8. Can I build a 50-story commercial skyscraper?**                     | **Domain restriction**               | **Pass** (Refused to answer)                               |
| **9. Who is the CEO of Indecimal?**                                      | **Information void test**            | **Pass** (Refused to answer)                               |
| 10. _How many critical checkpoints are in the quality assurance system?_ | **Extract exact number**             | **Pass** (Successfully returns 445+)                       |

**Future Considerations (Self-Correction):** The system proved robust against hallucination parameters when the answer wasn't present; however, strict prompt instruction adherence originally dropped heavily when asked to recall past conversational intent ("What was my last question?"). I solved this natively in LangChain by engineering a secondary context layer for memory injection, allowing the bot to interact with the user via `Previous Chat History` variables without losing factual domain mapping.
