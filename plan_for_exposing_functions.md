# Plan for Exposing Important Functions in Main Files

## Information Gathered
- 'org/main.py' currently has CLI commands for ingestion, Q&A, summarization, stats, export, reset, and interactive modes.
- Supporting modules (auth_manager.py, chroma_manager.py, document_processor.py, pubmed.py, qa_engine.py, rbac.py, settings.py) have important functions not fully exposed in main.py or launch.py.
- streamlit_app.py integrates many features with UI and authentication but is not exposed via CLI.
- demo.py has demo functions not exposed as CLI commands.

## Plan

### Files to Edit

- org/main.py
  - Add CLI commands for:
    - User authentication (login, logout, session validation)
    - User permission checks and role info
    - Vector store management: delete documents, update documents, backup/export collection
    - Document processing utilities: cleaning, chunking, key info extraction (optional)
    - Demo functions: test Euri connection, demo ingestion, demo Q&A, demo analytics
  - Integrate RBAC permission checks in CLI commands where applicable

- org/launch.py
  - Optionally add CLI commands or flags to expose environment validation and Streamlit app launch

- org/demo.py
  - Optionally convert demo functions to CLI commands or tests for easier access

- org/streamlit_app.py
  - Review for any functions not exposed in main.py or launch.py (no immediate edits unless requested)

## Dependent Files to Edit
- None initially, changes mostly additive in main.py and possibly launch.py

## Followup Steps
- Implement CLI commands in main.py as per plan
- Test new CLI commands for functionality and permission enforcement
- Optionally add tests for demo functions
- Verify integration with existing modules and settings
