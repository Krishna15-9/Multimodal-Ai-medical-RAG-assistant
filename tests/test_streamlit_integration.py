import unittest
import streamlit as st
import streamlit_app

class TestStreamlitIntegration(unittest.TestCase):
    def setUp(self):
        # Setup can include initializing session state or mocks if needed
        pass

    def test_login_page_renders(self):
        # Test that login page renders without error
        st.session_state.authenticated_user = None
        try:
            streamlit_app.login_page()
        except Exception as e:
            self.fail(f"login_page() raised Exception unexpectedly: {e}")

    def test_initialize_components(self):
        # Test initialization of QA engine and OpenRouter client
        st.session_state.qa_engine = None
        result = streamlit_app.initialize_components()
        self.assertTrue(result)
        self.assertIsNotNone(st.session_state.qa_engine)

    def test_get_collection_stats(self):
        # Test retrieval of collection stats
        st.session_state.collection_stats = None
        stats = streamlit_app.get_collection_stats()
        self.assertIsInstance(stats, dict)

    def test_research_and_ingest_page(self):
        # Test that research and ingest page renders without error
        try:
            streamlit_app.research_and_ingest_page()
        except Exception as e:
            self.fail(f"research_and_ingest_page() raised Exception unexpectedly: {e}")

    def test_ask_questions_page(self):
        # Test that ask questions page renders without error
        st.session_state.qa_engine = streamlit_app.QAEngine()
        st.session_state.collection_stats = {"total_documents": 1}
        try:
            streamlit_app.ask_questions_page()
        except Exception as e:
            self.fail(f"ask_questions_page() raised Exception unexpectedly: {e}")

    def test_analytics_page(self):
        # Test that analytics page renders without error
        st.session_state.collection_stats = {"total_documents": 1}
        try:
            streamlit_app.analytics_page()
        except Exception as e:
            self.fail(f"analytics_page() raised Exception unexpectedly: {e}")

    def test_settings_page(self):
        # Test that settings page renders without error
        try:
            streamlit_app.settings_page()
        except Exception as e:
            self.fail(f"settings_page() raised Exception unexpectedly: {e}")

if __name__ == "__main__":
    unittest.main()
