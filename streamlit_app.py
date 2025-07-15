
"""
Healthcare Q&A Tool - Professional Streamlit Application

An enterprise-grade, AI-powered healthcare research platform designed for
 to analyze intermittent fasting and metabolic
disorder research through advanced RAG (Retrieval-Augmented Generation)
architecture.

Technical Stack:
- Frontend: Streamlit with custom CSS and interactive components
- Backend: Modular Python architecture with dependency injection
- AI/ML: Euri AI integration with healthcare-optimized prompting
- Vector DB: ChromaDB with semantic search capabilities
- Data Source: PubMed API with intelligent filtering and processing
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import logging
# streamlit_app.py (updated)
# Replace:
# from src.llm import OpenRouterClient
# With:
from src.llm.openrouter_client import OpenRouterClient

# Later in the code, replace OpenRouterClient() with OpenRouterClient()
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.auth import AuthManager, RBACManager, Role
from src.config import get_settings
from src.llm.openrouter_client import OpenRouterClient
from src.processing import DocumentProcessor
from src.qa_system import QAEngine
from src.vector_store import ChromaManager

# Page configuration
st.set_page_config(
    page_title=" AI medical Health assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f8ff, #e6f3ff);
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4, #2e86ab);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #2e86ab, #1f77b4);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'qa_engine' not in st.session_state:
    st.session_state.qa_engine = None
if 'collection_stats' not in st.session_state:
    st.session_state.collection_stats = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'auth_manager' not in st.session_state:
    st.session_state.auth_manager = AuthManager()
if 'rbac_manager' not in st.session_state:
    st.session_state.rbac_manager = RBACManager()
if 'authenticated_user' not in st.session_state:
    st.session_state.authenticated_user = None
if 'session_token' not in st.session_state:
    st.session_state.session_token = None

def initialize_components():
    """Initialize the application components."""
    try:
        if st.session_state.qa_engine is None:
            with st.spinner("Initializing Healthcare Q&A System..."):
                st.session_state.qa_engine = QAEngine()
                
                # Test Euri connection
                euri_client = OpenRouterClient()
                if not euri_client.test_connection():
                    st.error("⚠️ Unable to connect to Openrouter AI. Please check your API key.")
                    return False
                    
        return True
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return False

def get_collection_stats():
    """Get and cache collection statistics."""
    if st.session_state.collection_stats is None:
        try:
            if st.session_state.qa_engine:
                st.session_state.collection_stats = st.session_state.qa_engine.get_collection_insights()
        except Exception as e:
            st.error(f"Error getting collection stats: {str(e)}")
            st.session_state.collection_stats = {"total_documents": 0}
    
    return st.session_state.collection_stats

def login_page():
    """Display login page with RBAC authentication."""
    st.markdown('<div class="main-header">🏥  AI medical Health assistant - Login</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #7f8c8d; font-size: 1.1rem;">Secure Access for Healthcare Professionals</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 Professional Login")

        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")

            col_login, col_demo = st.columns(2)

            with col_login:
                login_button = st.form_submit_button("🔑 Login", use_container_width=True)

            with col_demo:
                demo_button = st.form_submit_button("👥 Demo Credentials", use_container_width=True)

        if login_button and username and password:
            user = st.session_state.auth_manager.authenticate_user(username, password)

            if user:
                st.session_state.authenticated_user = user
                st.session_state.session_token = st.session_state.auth_manager.create_session_token(user)
                st.success(f"✅ Welcome, {user.full_name}!")
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Please try again.")

        if demo_button:
            st.markdown("### 👥 Demo User Credentials")

            demo_creds = st.session_state.auth_manager.get_demo_credentials()

            for role, creds in demo_creds.items():
                with st.expander(f"🎭 {creds['role']} - {creds['username']}"):
                    st.markdown(f"**Username:** `{creds['username']}`")
                    st.markdown(f"**Password:** `{creds['password']}`")
                    st.markdown(f"**Description:** {creds['description']}")

                    if st.button(f"Quick Login as {creds['role']}", key=f"quick_{role}"):
                        user = st.session_state.auth_manager.authenticate_user(
                            creds['username'],
                            creds['password']
                        )
                        if user:
                            st.session_state.authenticated_user = user
                            st.session_state.session_token = st.session_state.auth_manager.create_session_token(user)
                            st.success(f"✅ Logged in as {user.full_name}")
                            st.rerun()

def main():
    """Main application function."""
    # Check authentication
    if not st.session_state.authenticated_user:
        login_page()
        return

    user = st.session_state.authenticated_user
    rbac = st.session_state.rbac_manager

    # Header with user info
    st.markdown('<div class="main-header">🏥 AI medical Health assistant</div>', unsafe_allow_html=True)

    # User info bar
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown(f'<p style="color: #7f8c8d;">Welcome, <strong>{user.full_name}</strong> ({user.role.title()})</p>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<p style="color: #7f8c8d; text-align: center;"></p>', unsafe_allow_html=True)
    with col3:
        # Remove "MediInsight Health Solutions" text next to logout button by only showing the button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.auth_manager.logout_user(st.session_state.session_token)
            st.session_state.authenticated_user = None
            st.session_state.session_token = None
            st.rerun()

    # Initialize components
    if not initialize_components():
        st.stop()

    # Build navigation menu based on permissions
    nav_options = []
    nav_icons = []

    if rbac.has_feature_access(user, "research_ingest"):
        nav_options.append("🔍 Research & Ingest")
        nav_icons.append("search")

    if rbac.has_feature_access(user, "ask_questions"):
        nav_options.append("💬 Ask Questions")
        nav_icons.append("chat-dots")

    if rbac.has_feature_access(user, "analytics_dashboard"):
        nav_options.append("📊 Analytics")
        nav_icons.append("bar-chart")

    if rbac.has_feature_access(user, "system_settings"):
        nav_options.append("⚙️ Settings")
        nav_icons.append("gear")

    # Add Medical Image Analysis section (no permission check as per user request)
    nav_options.append("🖼️ Medical Image Analysis")
    nav_icons.append("image")

    # Navigation menu
    if nav_options:
        selected = option_menu(
            menu_title=None,
            options=nav_options,
            icons=nav_icons,
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1f77b4", "font-size": "18px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "#1f77b4"},
            }
        )

        # Route to different pages based on permissions
        if selected == "🔍 Research & Ingest" and rbac.has_feature_access(user, "research_ingest"):
            research_and_ingest_page()
        elif selected == "💬 Ask Questions" and rbac.has_feature_access(user, "ask_questions"):
            ask_questions_page()
        elif selected == "📊 Analytics" and rbac.has_feature_access(user, "analytics_dashboard"):
            analytics_page()
        elif selected == "⚙️ Settings" and rbac.has_feature_access(user, "system_settings"):
            settings_page()
        elif selected == "🖼️ Medical Image Analysis":
            medical_image_analysis_page()
    else:
        st.error("❌ No accessible features for your role. Please contact an administrator.")

def research_and_ingest_page():
    """Research and document ingestion page."""
    st.markdown('<div class="sub-header"> Research & Document Ingestion</div>', unsafe_allow_html=True)
    
    # Sidebar for search parameters
    with st.sidebar:
        st.markdown("### 🔍 Search Parameters")
        
        search_term = st.text_input(
            "Search Term",
            placeholder="e.g., intermittent fasting obesity",
            help="Enter keywords to search PubMed articles"
        )
        
        max_results = st.slider(
            "Maximum Results",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Number of articles to retrieve"
        )
        
        reset_collection = st.checkbox(
            "Reset Collection",
            help="Clear existing documents before ingesting new ones"
        )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            date_range = st.selectbox(
                "Publication Date Range",
                ["All years", "Last 5 years", "Last 10 years", "2020-2024"],
                help="Filter articles by publication date"
            )
            
            article_types = st.multiselect(
                "Article Types",
                ["Clinical Trial", "Randomized Controlled Trial", "Review", "Meta-Analysis"],
                help="Filter by specific article types"
            )
        
        # Ingest button
        if st.button("🚀 Start Ingestion", type="primary", use_container_width=True):
            if search_term:
                ingest_documents(search_term, max_results, reset_collection, date_range, article_types)
            else:
                st.error("Please enter a search term")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📋 Ingestion Instructions")
        st.markdown("""
        **Follow these steps to ingest research articles:**
        
        1. **Enter Search Terms**: Use relevant keywords related to your research topic
        2. **Set Parameters**: Choose the number of articles and other filters
        3. **Start Ingestion**: Click the button to begin retrieving and processing articles
        4. **Monitor Progress**: Watch the progress indicators and results
        
        **Recommended Search Terms for Healthcare Research:**
        - `intermittent fasting obesity`
        - `time-restricted eating diabetes`
        - `metabolic syndrome treatment`
        - `weight loss interventions`
        """)
        
        # Display recent ingestion results if available
        if 'last_ingestion_results' in st.session_state:
            results = st.session_state.last_ingestion_results
            st.markdown("### 📊 Last Ingestion Results")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Articles Found", results.get('articles_found_in_pubmed', 0))
            with col_b:
                st.metric("Articles Processed", results.get('total_articles_processed', 0))
            with col_c:
                st.metric("Added to Vector Store", results.get('articles_added_to_vector_store', 0))
    
    with col2:
        st.markdown("### 📈 Collection Overview")
        stats = get_collection_stats()
        
        # Display current collection stats
        st.metric("Total Documents", stats.get('total_documents', 0))
        st.metric("Unique Journals", stats.get('unique_journals', 0))
        
        # Publication years chart if available
        if stats.get('publication_years'):
            years = stats['publication_years'][-10:]  # Last 10 years
            if years:
                fig = px.bar(
                    x=years,
                    y=[1] * len(years),
                    title="Recent Publication Years",
                    labels={'x': 'Year', 'y': 'Count'}
                )
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

def ingest_documents(search_term: str, max_results: int, reset_collection: bool, date_range: str, article_types: List[str]):
    """Handle document ingestion process."""
    try:
        processor = DocumentProcessor()

        # Create progress containers
        progress_container = st.container()
        results_container = st.container()

        with progress_container:
            st.markdown("### 🔄 Ingestion Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Update progress
            status_text.text("🔍 Searching PubMed articles...")
            progress_bar.progress(20)

            # Prepare search parameters
            search_kwargs = {}
            if date_range != "All years":
                if date_range == "Last 5 years":
                    search_kwargs['date_range'] = "2019:2024"
                elif date_range == "Last 10 years":
                    search_kwargs['date_range'] = "2014:2024"
                elif date_range == "2020-2024":
                    search_kwargs['date_range'] = "2020:2024"

            if article_types:
                search_kwargs['article_types'] = article_types

            status_text.text("📥 Processing and ingesting articles...")
            progress_bar.progress(60)

            # Run ingestion pipeline
            results = processor.search_and_ingest_pipeline(
                search_term=search_term,
                max_results=max_results,
                reset_collection=reset_collection,
                **search_kwargs
            )

            progress_bar.progress(100)
            status_text.text("✅ Ingestion completed!")

            # Store results in session state
            st.session_state.last_ingestion_results = results
            st.session_state.collection_stats = None  # Reset to refresh stats

        with results_container:
            if results.get('success'):
                st.markdown('<div class="success-message">✅ Ingestion completed successfully!</div>', unsafe_allow_html=True)

                # Display detailed results
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("PubMed Articles Found", results.get('articles_found_in_pubmed', 0))
                with col2:
                    st.metric("Articles Processed", results.get('total_articles_processed', 0))
                with col3:
                    st.metric("High-Relevance Articles", results.get('high_relevance_articles', 0))
                with col4:
                    st.metric("Added to Vector Store", results.get('articles_added_to_vector_store', 0))

                # Show collection stats
                if results.get('collection_stats'):
                    stats = results['collection_stats']
                    st.markdown("### 📊 Updated Collection Statistics")
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Total Documents in Collection", stats.get('total_documents', 0))
                    with col_b:
                        st.metric("Unique Journals", stats.get('unique_journals', 0))
            else:
                st.markdown(f'<div class="error-message">❌ Ingestion failed: {results.get("error", "Unknown error")}</div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown(f'<div class="error-message">❌ Error during ingestion: {str(e)}</div>', unsafe_allow_html=True)

def ask_questions_page():
    """Q&A interface page."""
    st.markdown('<div class="sub-header"> Ask Questions</div>', unsafe_allow_html=True)

    # Check if collection has documents
    stats = get_collection_stats()
    if stats.get('total_documents', 0) == 0:
        st.markdown('<div class="warning-message">⚠️ No documents in the collection. Please ingest some articles first using the Research & Ingest page.</div>', unsafe_allow_html=True)
        return

    # Sidebar for Q&A settings
    with st.sidebar:
        st.markdown("### ⚙️ Q&A Settings")

        max_docs = st.slider(
            "Max Documents to Retrieve",
            min_value=3,
            max_value=15,
            value=5,
            help="Number of relevant documents to use for answering"
        )

        min_relevance = st.slider(
            "Minimum Relevance Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,  # Start with 0 to get any results
            step=0.1,
            help="Minimum relevance score for document retrieval (0 = no filter)"
        )

        include_sources = st.checkbox(
            "Include Sources",
            value=True,
            help="Show source information with answers"
        )

        # Quick question suggestions
        st.markdown("### 💡 Suggested Questions")
        suggestions = [
            "What are the benefits of intermittent fasting for obesity?",
            "How does time-restricted eating affect diabetes?",
            "What are the side effects of intermittent fasting?",
            "Which intermittent fasting protocol is most effective?",
            "How does intermittent fasting impact metabolic syndrome?"
        ]

        for suggestion in suggestions:
            if st.button(f"💭 {suggestion[:50]}...", key=f"suggest_{hash(suggestion)}", use_container_width=True):
                st.session_state.current_question = suggestion

    # Main Q&A interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🤔 Ask Your Question")

        # Question input
        question = st.text_area(
            "Your Question",
            value=st.session_state.get('current_question', ''),
            placeholder="Ask any question about the research articles in the collection...",
            height=100,
            key="question_input"
        )

        # Ask button
        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_button = st.button("🔍 Get Answer", type="primary", use_container_width=True)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.current_question = ""
                st.rerun()

        # Process question
        if ask_button and question.strip():
            with st.spinner("🤖 Analyzing research and generating answer..."):
                try:
                    response = st.session_state.qa_engine.ask_question(
                        question=question,
                        n_documents=max_docs,
                        min_relevance=min_relevance,
                        include_sources=include_sources
                    )

                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'response': response,
                        'timestamp': time.time()
                    })

                    # Display answer
                    display_qa_response(response, include_sources)

                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error generating answer: {str(e)}</div>', unsafe_allow_html=True)

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("### 📝 Recent Questions & Answers")

            # Show last 3 Q&As
            for i, chat in enumerate(reversed(st.session_state.chat_history[-3:])):
                with st.expander(f"Q: {chat['question'][:80]}...", expanded=(i == 0)):
                    display_qa_response(chat['response'], include_sources)

    with col2:
        st.markdown("### 📊 Collection Info")

        # Display collection statistics
        st.metric("Total Documents", stats.get('total_documents', 0))
        st.metric("Unique Journals", stats.get('unique_journals', 0))

        # Document analysis if available
        if stats.get('document_analysis'):
            analysis = stats['document_analysis']

            # Study types distribution
            if analysis.get('study_types'):
                st.markdown("#### Study Types")
                study_data = analysis['study_types']
                fig = px.pie(
                    values=list(study_data.values()),
                    names=[name.replace('_', ' ').title() for name in study_data.keys()],
                    title="Study Types Distribution"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Average relevance
            avg_relevance = analysis.get('average_relevance', 0)
            st.metric("Average Relevance Score", f"{avg_relevance:.2f}")

def display_qa_response(response: Dict, include_sources: bool):
    """Display Q&A response in a formatted way."""
    # Answer
    st.markdown("#### 🤖 Answer")
    st.markdown(response['answer'])

    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence = response.get('confidence', 0)
        color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
        st.markdown(f"**Confidence:** <span style='color: {color}'>{confidence:.2f}</span>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"**Sources Used:** {response.get('sources_count', 0)}")
    with col3:
        st.markdown(f"**Model:** {response.get('model_used', 'Unknown')}")

    # Sources
    if include_sources and response.get('sources'):
        st.markdown("#### 📚 Sources")
        for i, source in enumerate(response['sources'], 1):
            with st.expander(f"📄 {i}. {source['title'][:80]}..."):
                st.markdown(f"**Authors:** {source['authors']}")
                st.markdown(f"**Journal:** {source['journal']} ({source['publication_date']})")
                st.markdown(f"**Study Type:** {source['study_type'].replace('_', ' ').title()}")
                st.markdown(f"**PMID:** {source['pmid']}")
                if source.get('doi'):
                    st.markdown(f"**DOI:** {source['doi']}")
                st.markdown(f"**Relevance Score:** {source.get('relevance_score', 0):.2f}")

    # Error handling
    if response.get('error'):
        st.markdown(f'<div class="warning-message">⚠️ {response["error"]}</div>', unsafe_allow_html=True)

def analytics_page():
    """Analytics and insights page."""
    st.markdown('<div class="sub-header"> Analytics & Insights</div>', unsafe_allow_html=True)

    stats = get_collection_stats()

    if stats.get('total_documents', 0) == 0:
        st.markdown('<div class="warning-message">⚠️ No documents in the collection. Please ingest some articles first.</div>', unsafe_allow_html=True)
        return

    # Overview metrics
    st.markdown("### 📈 Collection Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Documents", stats.get('total_documents', 0))
    with col2:
        st.metric("Unique Journals", stats.get('unique_journals', 0))
    with col3:
        pub_years = stats.get('publication_years', [])
        year_range = f"{min(pub_years)}-{max(pub_years)}" if pub_years else "N/A"
        st.metric("Publication Range", year_range)
    with col4:
        article_types = len(stats.get('article_types', []))
        st.metric("Article Types", article_types)

    # Detailed analytics
    if stats.get('document_analysis'):
        analysis = stats['document_analysis']

        col_left, col_right = st.columns(2)

        with col_left:
            # Study types distribution
            if analysis.get('study_types'):
                st.markdown("#### 🔬 Study Types Distribution")
                study_data = analysis['study_types']

                fig = px.bar(
                    x=list(study_data.values()),
                    y=[name.replace('_', ' ').title() for name in study_data.keys()],
                    orientation='h',
                    title="Number of Studies by Type",
                    color=list(study_data.values()),
                    color_continuous_scale="Blues"
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Research focus areas
            if analysis.get('research_focus_areas'):
                st.markdown("#### 🎯 Research Focus Areas")
                focus_data = analysis['research_focus_areas']

                fig = px.pie(
                    values=list(focus_data.values()),
                    names=[name.replace('_', ' ').title() for name in focus_data.keys()],
                    title="Research Focus Distribution"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Publication timeline
            if analysis.get('publication_years'):
                st.markdown("#### 📅 Publication Timeline")
                year_data = analysis['publication_years']

                # Sort years and create timeline
                sorted_years = sorted(year_data.items())
                years = [item[0] for item in sorted_years]
                counts = [item[1] for item in sorted_years]

                fig = px.line(
                    x=years,
                    y=counts,
                    title="Publications Over Time",
                    markers=True
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            # Top journals
            if analysis.get('journals'):
                st.markdown("#### 📚 Top Journals")
                journal_data = analysis['journals']

                # Get top 10 journals
                top_journals = sorted(journal_data.items(), key=lambda x: x[1], reverse=True)[:10]

                if top_journals:
                    fig = px.bar(
                        x=[item[1] for item in top_journals],
                        y=[item[0][:30] + "..." if len(item[0]) > 30 else item[0] for item in top_journals],
                        orientation='h',
                        title="Articles by Journal",
                        color=[item[1] for item in top_journals],
                        color_continuous_scale="Greens"
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            # Quality metrics
            st.markdown("#### ⭐ Quality Metrics")
            avg_relevance = analysis.get('average_relevance', 0)

            # Create gauge chart for average relevance
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = avg_relevance,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Average Relevance Score"},
                delta = {'reference': 0.5},
                gauge = {
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "lightgray"},
                        {'range': [0.3, 0.7], 'color': "yellow"},
                        {'range': [0.7, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.8
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Export options
    st.markdown("### 💾 Export Options")
    col_exp1, col_exp2 = st.columns(2)

    with col_exp1:
        if st.button("📊 Export Analytics Report", use_container_width=True):
            # Generate analytics report
            report_data = {
                "collection_stats": stats,
                "export_timestamp": time.time(),
                "total_documents": stats.get('total_documents', 0)
            }

            st.download_button(
                label="📥 Download Report (JSON)",
                data=str(report_data),
                file_name=f"healthcare_qa_analytics_{int(time.time())}.json",
                mime="application/json"
            )

    with col_exp2:
        if st.button("🗂️ Export Collection Data", use_container_width=True):
            try:
                chroma_manager = st.session_state.get('chroma_manager')
                if chroma_manager is None:
                    chroma_manager = ChromaManager()
                    st.session_state.chroma_manager = chroma_manager
                export_path = f"collection_export_{int(time.time())}.json"

                if chroma_manager.export_collection(export_path):
                    st.success(f"✅ Collection exported to {export_path}")
                else:
                    st.error("❌ Export failed")
            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")

def settings_page():
    """Settings and configuration page."""
    st.markdown('<div class="sub-header">⚙️ Settings & Configuration</div>', unsafe_allow_html=True)

    # API Configuration
    st.markdown("### 🔑 API Configuration")

    try:
        settings = get_settings()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Euri AI Settings")

            # Test API connection
            if st.button("🔍 Test Euri Connection", use_container_width=True):
                try:
                    euri_client = OpenRouterClient()
                    if euri_client.test_connection():
                        st.success("✅ Openrouter connection successful!")
                    else:
                        st.error("❌ Openrouter AI connection failed!")
                except Exception as e:
                    st.error(f"❌ Connection test error: {str(e)}")

            # Display current settings (masked)
            st.text_input("API Key", value="*" * 20, disabled=True, help="API key is loaded from environment")
           # st.text_input("Base URL", value=settings.euri_base_url, disabled=True)
            st.text_input("Model", value=settings.llm_model, disabled=True)

        with col2:
            st.markdown("#### Collection Settings")
            st.text_input("Collection Name", value=settings.chroma_collection_name, disabled=True)
            st.text_input("Embedding Model", value=settings.embedding_model, disabled=True)
            st.number_input("Max Articles per Search", value=settings.max_articles_per_search, disabled=True)

    except Exception as e:
        st.error(f"❌ Error loading settings: {str(e)}")

    # Collection Management
    st.markdown("### 🗄️ Collection Management")

    col_mgmt1, col_mgmt2, col_mgmt3 = st.columns(3)

    with col_mgmt1:
        if st.button("📊 Refresh Stats", use_container_width=True):
            st.session_state.collection_stats = None
            st.rerun()

    with col_mgmt2:
        if st.button("🗑️ Reset Collection", use_container_width=True, type="secondary"):
            if st.checkbox("⚠️ I understand this will delete all documents"):
                try:
                    chroma_manager = st.session_state.get('chroma_manager')
                    if chroma_manager is None:
                        chroma_manager = ChromaManager()
                        st.session_state.chroma_manager = chroma_manager
                    if chroma_manager.reset_collection():
                        st.success("✅ Collection reset successfully!")
                        st.session_state.collection_stats = None
                        st.session_state.chat_history = []
                    else:
                        st.error("❌ Reset failed!")
                except Exception as e:
                    st.error(f"❌ Reset error: {str(e)}")

    with col_mgmt3:
        if st.button("💾 Backup Collection", use_container_width=True):
            try:
                chroma_manager = st.session_state.get('chroma_manager')
                if chroma_manager is None:
                    chroma_manager = ChromaManager()
                    st.session_state.chroma_manager = chroma_manager
                backup_path = f"backup_{int(time.time())}.json"

                if chroma_manager.export_collection(backup_path):
                    st.success(f"✅ Backup created: {backup_path}")
                else:
                    st.error("❌ Backup failed!")
            except Exception as e:
                st.error(f"❌ Backup error: {str(e)}")

    # System Information
    st.markdown("### ℹ️ System Information")

    info_col1, info_col2 = st.columns(2)

    with info_col1:
        st.markdown("**Application Version:** 1.0.0")
        st.markdown("**Framework:** Streamlit")
        st.markdown("**Vector Store:** Chroma DB")

    with info_col2:
        st.markdown("**LLM Provider:** Openrouter")
        st.markdown("**Data Source:** PubMed")
        st.markdown("**Embedding Model:** Sentence Transformers")

    # Help and Documentation
    st.markdown("### 📖 Help & Documentation")

    with st.expander("🔧 How to Use This Tool"):
        st.markdown("""
        **Getting Started:**
        1. Go to the "Research & Ingest" page
        2. Enter search terms related to your research topic
        3. Click "Start Ingestion" to retrieve and process articles
        4. Use the "Ask Questions" page to query the ingested articles

        **Best Practices:**
        - Use specific, relevant search terms
        - Start with 50-100 articles for initial testing
        - Use the analytics page to understand your collection
        - Regularly backup your collection data
        """)

    with st.expander("❓ Troubleshooting"):
        st.markdown("""
        **Common Issues:**
        - **No API connection:** Check your Openrouter key in the .env file
        - **No search results:** Try broader search terms or increase max results
        - **Poor answer quality:** Ensure you have relevant, high-quality articles
        - **Slow performance:** Reduce the number of documents retrieved per query
        """)

def startup_check():
    """Perform startup checks and display system status."""
    st.sidebar.markdown("### 🔧 System Status")

    # Check environment configuration
    try:
        settings = get_settings()
        st.sidebar.success("✅ Configuration loaded")

        # Test Euri connection
        try:
            euri_client = OpenRouterClient()
            if euri_client.test_connection():
                st.sidebar.success("✅ Openrouter connected")
            else:
                st.sidebar.error("❌ Openrouter connection failed")
        except Exception as e:
            st.sidebar.error(f"❌ Euri AI error: {str(e)[:50]}...")

        # Check vector store
        if 'chroma_manager' not in st.session_state:
            try:
                st.session_state.chroma_manager = ChromaManager()
                st.session_state.vectorstore_initialized = True
            except Exception as e:
                st.session_state.vectorstore_initialized = False
                st.sidebar.warning(f"⚠️ Vector store initialization failed: {str(e)[:50]}...")
        else:
            # Already initialized
            pass

        if st.session_state.get('vectorstore_initialized', False):
            try:
                stats = st.session_state.chroma_manager.get_collection_stats()
                doc_count = stats.get('total_documents', 0)
                st.sidebar.info(f"📚 {doc_count} documents in collection")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Vector store stats retrieval failed: {str(e)[:50]}...")
        else:
            st.sidebar.warning("⚠️ Vector store not initialized")

    except Exception as e:
        st.sidebar.error("❌ Configuration error")
        st.sidebar.error("Please check your .env file")

import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

# Set your API Key (Replace with your actual key or load from environment)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyCUPYFW5ESk1YIwSjwpGd3jZJJ7oPAX4_s")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize the Medical Agent once globally
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# Medical Analysis Query defined once globally
query = """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.).
- Specify anatomical region and positioning.
- Evaluate image quality and technical adequacy.

### 2. Key Findings
- Highlight primary observations systematically.
- Identify potential abnormalities with detailed descriptions.
- Include measurements and densities where relevant.

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level.
- List differential diagnoses ranked by likelihood.
- Support each diagnosis with observed evidence.
- Highlight critical/urgent findings.

### 4. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language.
- Avoid medical jargon or provide easy definitions.
- Include relatable visual analogies.

### 5. Research Context
- Use DuckDuckGo search to find recent medical literature.
- Search for standard treatment protocols.
- Provide 2-3 key references supporting the analysis.

Ensure a structured and medically accurate response using clear markdown formatting.
"""

def medical_image_analysis_page():
    """Medical Image Analysis page using Google Gemini 2.5 Pro API."""
    st.markdown(
        """
        <iframe
        src="https://lucifer7210-multimodal-llm.hf.space"
        frameborder="0"
        width="100%"
        height="1000"
        ></iframe>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    # Add startup check to sidebar
    startup_check()

    # Run main application
    main()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p><strong> AI medical Health assistant v1.0</strong></p>
            
        Advancing healthcare research through AI-powered literature analysis
        </div>
        """,
        unsafe_allow_html=True
    )
