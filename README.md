# 🏥 Healthcare Q&A Tool: Enterprise RAG Platform

> **An advanced Retrieval-Augmented Generation (RAG) system engineered for healthcare research intelligence, featuring microservices architecture, semantic vector search, and AI-powered knowledge synthesis.**

&#x20; &#x20;

## 🌟 Executive Summary

This enterprise-grade healthcare research platform demonstrates advanced software engineering principles through a sophisticated RAG architecture. Built for MediInsight Health Solutions, it showcases expertise in **distributed systems design**, **AI/ML integration**, **semantic search optimization**, and **production-ready software development**.

## 🏠 Architecture Overview

- **Microservices-based** with modular components for ingestion, processing, and LLM inference.
- **ChromaDB** for high-performance vector search.
- **Streamlit** for an enterprise-grade front-end interface.
- **Role-Based Access Control** with JWT authentication.

## 🤖 AI Integration

```python
# Professional LLM integration with OpenRouter API
class OpenRouterClient:
    def __init__(self):
        self.client = OpenRouterAPI(
            api_key=self.settings.openrouter_api_key,
            model="mistral:instruct"  # Example model
        )

    def generate_healthcare_response(self, query: str, context: str) -> str:
        prompt = self._convert_messages_to_prompt([
            {"role": "system", "content": self._get_healthcare_system_prompt()},
            {"role": "user", "content": self._create_healthcare_prompt(query, context)}
        ])

        return self.client.generate_completion(
            prompt=prompt,
            temperature=0.1,
            max_tokens=500
        )
```

## 📄 Environment Configuration

```bash
# Required: OpenRouter Integration
OPENROUTER_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=mistral:instruct

# Authentication & Security
ENABLE_AUTHENTICATION=true
JWT_SECRET_KEY=your_jwt_secret_key_here_change_in_production
SESSION_TIMEOUT_HOURS=8

# Performance Tuning
MAX_ARTICLES_PER_SEARCH=100
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
VECTOR_DIMENSION=384

# System Configuration
LOG_LEVEL=INFO
CHROMA_COLLECTION_NAME=healthcare_articles
PUBMED_RATE_LIMIT_DELAY=1.0
```

## 🚀 RAG Pipeline Architecture

```python
class AdvancedRAGPipeline:
    def __init__(self):
        self.retriever = SemanticRetriever(embedding_model="sentence-transformers/all-MiniLM-L6-v2")
        self.reranker = CrossEncoderReranker(model="ms-marco-MiniLM-L-12-v2")
        self.generator = OpenRouterClient()
        self.context_optimizer = ContextWindowOptimizer(max_tokens=4000)

    async def process_query(self, query: str) -> RAGResponse:
        candidates = await self.retriever.retrieve(query, top_k=20)
        reranked = await self.reranker.rerank(query, candidates, top_k=5)
        optimized_context = self.context_optimizer.optimize(reranked)

        response = await self.generator.generate_healthcare_response(
            query=query,
            context=optimized_context,
            confidence_threshold=0.7
        )

        return RAGResponse(
            answer=response.text,
            confidence=response.confidence,
            sources=reranked,
            processing_time=response.latency
        )
```

## 🔧 System Validation

```bash
# Run comprehensive system test
python demo.py

# Expected output: All components validated
# ✅ OpenRouter Connection
# ✅ Document Processing
# ✅ Vector Search
# ✅ Q&A Generation
```

## 📊 Monitoring & Observability

- Logging and metrics capture OpenRouter latency, token usage, and error rates.

## 🌟 Technical Leadership Demonstration

- Uses OpenRouter API Client for LLM inference.
- All legacy references to Euri SDK have been removed.

## 📈 Advanced Capabilities

- **Secure RAG with role-aware responses.**
- **Vector search with semantic filtering.**
- **Configurable prompting for medical QA.**
- **Interactive dashboard and API-based automation.**

## 🖼️ New Feature: Medical Image Analysis

- Integrated a new Medical Image Analysis section in the Streamlit app.
- Embeds an AI-powered medical image analysis tool using Google Gemini 2.5 Pro API.
- Allows users to upload medical images (X-ray, MRI, CT, Ultrasound, etc.) for detailed AI-driven analysis.
- Provides diagnostic assessments, patient-friendly explanations, and research context.
- Utilizes DuckDuckGo tools for up-to-date medical literature references.
- Accessible via the "Medical Image Analysis" tab in the main navigation menu.

