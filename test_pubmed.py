from src.data_retrieval import PubMedRetriever

def test_pubmed():
    retriever = PubMedRetriever()
    
    # Try normal search
    pmids = retriever.search_articles("cancer", 5)
    
    # Fallback to direct search if normal fails
    if not pmids:
        print("Normal search failed, trying direct search...")
        pmids = retriever._direct_search("cancer", 5)
    
    print(f"Found PMIDs: {pmids}")
    
    if pmids:
        articles = retriever.fetch_articles(pmids[:2])
        print(f"First article: {articles[0]['title'] if articles else 'None'}")
    else:
        print("""
        All search methods failed. Please check:
        1. Internet connection
        2. https://www.ncbi.nlm.nih.gov/books/NBK25497/ for API status
        3. Firewall/antivirus settings
        """)

if __name__ == "__main__":
    test_pubmed()