# DocSense — Document Intelligence (Declarative DAG Pipeline) <div align="center"> [![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![MLflow](https://img.shields.io/badge/MLflow-Registry-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-blue.svg?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) </div> > **Multimodal document intelligence and grounded Q&A system architected as an explicit Declarative Directed Acyclic Graph (DAG) pipeline with topological stage scheduling, node result caching, and hybrid lexical-dense retrieval.** --- ## 🏛️ Architecture Pattern: Declarative DAG Pipeline Architecture Complex retrieval-augmented generation (RAG) systems suffer when implemented as procedural monolithic scripts. Changes to embedding models, chunking strategies, or fusion algorithms cause cascading side effects. `docsense` decomposes the document ingestion and Q&A lifecycle into a **Declarative Directed Acyclic Graph (DAG)** of pure, typed compute nodes: ```mermaid
> **Note:** This is a portfolio project demonstrating software engineering patterns and ML concepts. Not intended for production use without further hardening. graph TD subgraph Input_Stage ["Query Input"] Q[Natural Language Question] end subgraph Retrieval_DAG ["Parallel Retrieval Stage"] DenseNode[DenseRetrievalNode<br/>Cosine Nearest Neighbor] BM25Node[BM25RetrievalNode<br/>Lexical BM25Okapi] end subgraph Fusion_DAG ["Rank Fusion & Context Assembly"] RRFNode[RRFMergeNode<br/>Reciprocal Rank Fusion k=60] CtxNode[ContextAssemblyNode<br/>Token Window Packing] end subgraph Synthesis_DAG ["Prompting & LLM Generation"] PromptNode[PromptFormatNode<br/>Template Variable Binding] LLMNode[LLMSynthesisNode<br/>Claude / Ollama / FakeProvider] end Q --> DenseNode Q --> BM25Node DenseNode --> RRFNode BM25Node --> RRFNode RRFNode --> CtxNode CtxNode --> PromptNode Q --> PromptNode PromptNode --> LLMNode
``` ### DAG Architecture Highlights
- **`dag/node.py`**: Protocol defining `DAGNode` and `NodeResult` with explicit dependencies and typed context passing.
- **`dag/graph.py`**: Topological graph runner implementing Kahn's algorithm with cycle detection and dependency validation.
- **`dag/nodes.py`**: Discrete pure execution nodes: - `DenseRetrievalNode`: Executes cosine semantic similarity over vector embeddings. - `BM25RetrievalNode`: Executes tokenized lexical frequency scoring. - `RRFMergeNode`: Performs non-parametric Reciprocal Rank Fusion ($k=60$). - `ContextAssemblyNode`: Enforces strict character and token window constraints. - `PromptFormatNode`: Injects retrieved snippets with page-level citations. - `LLMSynthesisNode`: Invokes LLM providers with streaming and batch contracts.
- **`dag/pipeline.py`**: High-level declarative pipeline factory `build_doc_qa_dag()` and execution runner. --- ## 📑 Core Methodologies & Retrieval Formulation ### 1. Document Extraction & OCR Pipeline
- Direct text extraction for digital PDFs via PyPDF.
- Automatic OCR fallback for scanned documents and images using Tesseract / PaddleOCR with deskewing and contrast preprocessing. ### 2. Hybrid Lexical-Dense Retrieval (RRF)
- Merges ranked lists using Reciprocal Rank Fusion: $$\text{RRF}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + r_m(d)}, \quad k = 60$$
- Eliminates score scale calibration issues between cosine distance and unbounded BM25 scores. ### 3. Provenance-Grounded Citations
- Generates verified answers with exact document IDs, page indices, and source snippets.
- Hot-swappable LLM backends supporting Anthropic Claude, local Ollama (Llama 3/Mistral), and deterministic mocks for offline CI. --- ## 🚀 Quickstart & Setup Guide ### 1. Prerequisites & Environment Setup
```bash
# Clone repository
git clone https://github.com/jackson-marcus/docsense.git
cd docsense # Install dependencies via uv
$env:UV_CACHE_DIR = "D:\ml-projects\.uv-cache"
uv sync --group dev
``` ### 2. Run Test Suite & Code Quality Checks
```bash
# Run unit & DAG pipeline tests
uv run pytest -q # Run ruff linter and formatting checks
uv run ruff check .
uv run ruff format --check .
``` ### 3. Launch Services Locally
```bash
# Start FastAPI REST API (listening on port :8010)
make api # Start interactive Streamlit document workspace (listening on port :8511)
make ui
``` --- ## 📂 Repository Layout ```
docsense/
├── configs/ # Configuration files (retrieval, chunking, LLM)
├── data/ # Document store and persistent indices
├── src/docsense/ # Core Python package
│ ├── dag/ # Declarative DAG pipeline (graph runner, node contracts, pipeline builder)
│ ├── ingestion/ # OCR and PDF loaders
│ ├── indexing/ # Text chunker, embeddings, and Chroma store
│ ├── retrieval/ # Dense and BM25 search engines with RRF fusion
│ ├── llm/ # LLM provider factory (Claude, Ollama, Fake)
│ ├── rag/ # RAG chain wrapping DAG execution
│ ├── api/ # FastAPI REST routes and streaming endpoints
│ └── ui/ # Streamlit interactive application
├── tests/ # Comprehensive Pytest suite covering DAG and RAG
├── docker-compose.yml # Multi-service container orchestration
├── Dockerfile # Container definition for API service
└── pyproject.toml # Pinned dependencies and tool configs
``` --- ## 👤 Author & Contact **Jackson Marcus**
- **Email:** [jackson.marcus.work@gmail.com](mailto:jackson.marcus.work@gmail.com)
- **Upwork:** [Jackson Marcus on Upwork](https://www.upwork.com/freelancers/~012235717501ad9c7b)
- **GitHub:** [@jackson-marcus](https://github.com/jackson-marcus) --- ## 👨‍💻 Author & Maintainer <div align="center"> ### **Jackson Marcus**
**Senior AI & Machine Learning Engineer**
*Building ML Systems, Agentic Architectures & Scalable Data Pipelines* [![GitHub Profile](https://img.shields.io/badge/GitHub-jackson--marcus-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/jackson-marcus)
[![Upwork Portfolio](https://img.shields.io/badge/Upwork-Top%20Rated%20Plus-14A800?style=for-the-badge&logo=upwork&logoColor=white)](https://www.upwork.com/freelancers/~012235717501ad9c7b)
[![Email Contact](https://img.shields.io/badge/Email-wajahatanees41%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:wajahatanees41@gmail.com) 📍 *Byron, GA, USA* </div>
