# Missing Important Functions in Main Files Compared to 'org/' Folder

## Overview
This report identifies important functions present in the original 'org/' folder files that are missing or not fully exposed in the main files of the project, primarily `org/main.py` and `org/launch.py`.

## Main Files Reviewed
- org/main.py
- org/launch.py

## Supporting Modules Reviewed
- auth_manager.py
- chroma_manager.py
- document_processor.py
- pubmed.py
- qa_engine.py
- rbac.py
- settings.py
- demo.py

## Missing Important Functions in Main Files

### 1. Authentication and Session Management (auth_manager.py)
- `authenticate_user(username, password)`
- `create_session_token(user)`
- `validate_token(token)`
- `get_user_permissions(user)`
- `has_permission(user, permission)`
- `logout_user(token)`
- Demo credentials management

### 2. Vector Store Management (chroma_manager.py)
- `delete_documents(document_ids)`
- `update_document(document_id, document, metadata)`
- `reset_collection()`
- `export_collection(output_path)`
- Detailed collection stats and metadata analysis

### 3. Document Processing Utilities (document_processor.py)
- `clean_text(text)`
- `extract_key_information(article)`
- `chunk_document(article)`
- `process_articles(articles)`
- `ingest_articles_to_vector_store(articles, reset_collection)`
- `search_and_ingest_pipeline(search_term, max_results, reset_collection)`

### 4. PubMed Retrieval (pubmed.py)
- `search_pubmed_articles(search_term, max_results)`
- `fetch_pubmed_abstracts(pmid_list)`

### 5. Q&A Engine Advanced Functions (qa_engine.py)
- `retrieve_relevant_documents(query, n_results, min_relevance_score)`
- `generate_answer(query, context_documents, include_sources)`
- `ask_question(question, n_documents, min_relevance, include_sources)`
- `get_research_summary(topic, max_documents)`
- `get_collection_insights()`
- Confidence scoring and source formatting

### 6. Role-Based Access Control (rbac.py)
- `has_permission(user, permission)`
- `has_feature_access(user, feature)`
- `get_user_permissions(user)`
- `get_accessible_features(user)`
- `get_role_description(role)`
- Bulk operation permission checks

### 7. Settings Management (settings.py)
- Environment variable loading and validation
- Directory management
- API key and LLM configuration access

### 8. Demo Functions (demo.py)
- `test_euri_connection()`
- `demo_document_ingestion()`
- `demo_qa_system()`
- `demo_analytics()`
- `main()` to run all demos

## Recommendations
- Expose the above important functions in `org/main.py` as CLI commands or API endpoints.
- Integrate RBAC and authentication checks in CLI commands.
- Add vector store management commands for delete, update, reset, export.
- Provide document processing utilities as CLI commands if needed.
- Add demo functions as CLI commands or test scripts for easier access.
- Optionally, enhance `org/launch.py` to expose environment validation and app launch commands.

This will ensure the main files fully leverage the capabilities of the supporting modules and provide comprehensive functionality.
