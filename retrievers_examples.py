"""
LangChain Retrievers - Complete Guide with Examples
====================================================

Retrievers are used to fetch relevant documents/information based on a query.
Each retriever has different use cases and requirements.
"""

from dotenv import load_dotenv
import os

load_dotenv()

# ════════════════════════════════════════════════════════════════════════════════
# 1. WIKIPEDIA RETRIEVER
# ════════════════════════════════════════════════════════════════════════════════
# Use Case: Search Wikipedia for general knowledge
# Pros: No API key needed, fast, reliable
# Cons: Limited to Wikipedia content only

def example_wikipedia_retriever():
    print("\n" + "="*70)
    print("1. WIKIPEDIA RETRIEVER")
    print("="*70)
    
    from langchain_community.retrievers import WikipediaRetriever
    
    # Create retriever instance
    retriever = WikipediaRetriever()
    
    # Invoke with query
    query = "What is the capital of India?"
    docs = retriever.invoke(query)
    
    # docs is a list of Document objects
    print(f"Query: {query}")
    print(f"Found {len(docs)} documents")
    if docs:
        print(f"\nFirst result (page_content preview):")
        print(docs[0].page_content[:500])  # First 500 chars
        print(f"\nMetadata: {docs[0].metadata}")

# ════════════════════════════════════════════════════════════════════════════════
# 2. ARXIV RETRIEVER
# ════════════════════════════════════════════════════════════════════════════════
# Use Case: Search academic papers on ArXiv
# Pros: No API key needed, huge academic database
# Cons: Technical/academic content only

def example_arxiv_retriever():
    print("\n" + "="*70)
    print("2. ARXIV RETRIEVER")
    print("="*70)
    
    from langchain_community.retrievers import ArxivRetriever
    
    # Create retriever instance
    retriever = ArxivRetriever()
    
    # Invoke with query
    query = "transformer neural networks"
    docs = retriever.invoke(query)
    
    # docs is a list of Document objects
    print(f"Query: {query}")
    print(f"Found {len(docs)} papers")
    if docs:
        print(f"\nFirst result (abstract preview):")
        print(docs[0].page_content[:500])
        print(f"\nMetadata (title, authors, date):")
        print(docs[0].metadata)

# ════════════════════════════════════════════════════════════════════════════════
# 3. COUNTRY WIKI DATA RETRIEVER
# ════════════════════════════════════════════════════════════════════════════════
# Use Case: Get information about countries
# Pros: Structured country data, no API key needed
# Cons: Limited to country information only

def example_country_retriever():
    print("\n" + "="*70)
    print("3. COUNTRY WIKI DATA RETRIEVER")
    print("="*70)
    
    from langchain_community.retrievers import WikipediaRetriever
    
    # You can use WikipediaRetriever for this too
    retriever = WikipediaRetriever()
    query = "India geography population"
    docs = retriever.invoke(query)
    
    print(f"Query: {query}")
    print(f"Found {len(docs)} documents")
    if docs:
        print(f"\nResult preview:")
        print(docs[0].page_content[:500])

# ════════════════════════════════════════════════════════════════════════════════
# 4. PUBMED RETRIEVER (Medical/Biomedical Papers)
# ════════════════════════════════════════════════════════════════════════════════
# Use Case: Search biomedical and life sciences literature
# Pros: No API key, huge medical database
# Cons: Medical/scientific content only

def example_pubmed_retriever():
    print("\n" + "="*70)
    print("4. PUBMED RETRIEVER (Medical Literature)")
    print("="*70)
    
    from langchain_community.retrievers import PubMedRetriever
    
    # Create retriever instance
    retriever = PubMedRetriever()
    
    # Invoke with medical query
    query = "COVID-19 vaccines"
    docs = retriever.invoke(query)
    
    print(f"Query: {query}")
    print(f"Found {len(docs)} papers")
    if docs:
        print(f"\nFirst result:")
        print(docs[0].page_content[:500])
        print(f"\nMetadata: {docs[0].metadata}")

# ════════════════════════════════════════════════════════════════════════════════
# 5. DUCKDUCKGO SEARCH (Using duckduckgo-search package directly)
# ════════════════════════════════════════════════════════════════════════════════
# Use Case: Web search without API key
# Pros: No API key needed, covers entire web
# Cons: May not be available in all LangChain versions

def example_duckduckgo_retriever():
    print("\n" + "="*70)
    print("5. DUCKDUCKGO SEARCH (Web Search - No API Key)")
    print("="*70)
    
    try:
        # Use the new 'ddgs' package (renamed from duckduckgo-search)
        from ddgs import DDGS
        
        ddgs = DDGS(timeout=10)
        query = "what is the capital of India?"
        results = list(ddgs.text(query, max_results=5))
        
        print(f"Query: {query}")
        print(f"Found {len(results)} results")
        if results:
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Title: {result.get('title', 'N/A')}")
                print(f"Link: {result.get('href', 'N/A')}")
                print(f"Summary: {result.get('body', 'N/A')[:300]}")
    except ImportError:
        print("⚠️  'ddgs' package not installed")
        print("Install: pip install ddgs")
    except Exception as e:
        print(f"Error: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# 6. GOOGLE SERPER API (Premium Google Search)
# ════════════════════════════════════════════════════════════════════════════════
# Use Case: High-quality Google search results
# Pros: Official Google results, fast
# Cons: Requires API key (paid service, but free tier available)

def example_google_serper_retriever():
    print("\n" + "="*70)
    print("6. GOOGLE SERPER API RETRIEVER (Premium Google Search)")
    print("="*70)
    
    serper_api_key = os.getenv("SERPER_API_KEY")
    
    if not serper_api_key:
        print("⚠️  SERPER_API_KEY not found in .env")
        print("Setup: https://serper.dev/ (100 free searches/month)")
        print("Add to .env: SERPER_API_KEY=your_key_here")
        return
    
    try:
        from langchain_community.retrievers import GoogleSerperAPIRetriever
        
        # Create retriever with API key
        retriever = GoogleSerperAPIRetriever(serper_api_key=serper_api_key)
        
        # Invoke with query
        query = "latest AI trends 2024"
        docs = retriever.invoke(query)
        
        print(f"Query: {query}")
        print(f"Found {len(docs)} results")
        if docs:
            print(f"\nFirst result:")
            print(docs[0].page_content[:500])
    except Exception as e:
        print(f"Error: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# 7. ARXIV PAPER RETRIEVER (Alternative - with more control)
# ════════════════════════════════════════════════════════════════════════════════
# Similar to ArxivRetriever but with more customization

def example_arxiv_advanced():
    print("\n" + "="*70)
    print("7. ARXIV RETRIEVER (Advanced)")
    print("="*70)
    
    from langchain_community.retrievers import ArxivRetriever
    
    # Create retriever with parameters
    retriever = ArxivRetriever(
        top_k_results=5,  # Number of results to return
        ARXIV_MAX_QUERY_LENGTH=300  # Max query length
    )
    
    query = "deep learning computer vision"
    docs = retriever.invoke(query)
    
    print(f"Query: {query}")
    print(f"Found {len(docs)} papers (limited to top 5)")
    if docs:
        for i, doc in enumerate(docs, 1):
            print(f"\nPaper {i}:")
            print(f"Title: {doc.metadata.get('Title', 'N/A')}")
            print(f"Authors: {doc.metadata.get('Authors', 'N/A')}")
            print(f"Summary: {doc.page_content[:300]}...")

# ════════════════════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ════════════════════════════════════════════════════════════════════════════════

RETRIEVER_COMPARISON = """
╔═══════════════════╦══════════════╦════════════════╦═══════════════════════════╗
║ Retriever         ║ API Key Req  ║ Use Case       ║ Coverage                  ║
╠═══════════════════╬══════════════╬════════════════╬═══════════════════════════╣
║ Wikipedia         ║ No           ║ General Info   ║ General Knowledge         ║
║ ArXiv             ║ No           ║ Academic Papers║ Scientific Papers         ║
║ PubMed            ║ No           ║ Medical Info   ║ Biomedical Literature     ║
║ DuckDuckGo        ║ No           ║ Web Search     ║ Entire Web                ║
║ Google Serper     ║ Yes (Free)   ║ Premium Search ║ Google Official Results   ║
║ Google Custom     ║ Yes          ║ Specific Sites ║ Domain-specific Search    ║
║ Bing Search       ║ Yes          ║ Web Search     ║ Bing Results              ║
╚═══════════════════╩══════════════╩════════════════╩═══════════════════════════╝
"""

# ════════════════════════════════════════════════════════════════════════════════
# MAIN - RUN ALL EXAMPLES
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  LANGCHAIN RETRIEVERS - COMPLETE GUIDE".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Run examples
    try:
        # Example 1: Wikipedia
        example_wikipedia_retriever()
    except Exception as e:
        print(f"❌ Wikipedia error: {e}")
    
    try:
        # Example 2: ArXiv
        example_arxiv_retriever()
    except Exception as e:
        print(f"❌ ArXiv error: {e}")
    
    try:
        # Example 4: PubMed
        example_pubmed_retriever()
    except Exception as e:
        print(f"❌ PubMed error: {e}")
    
    try:
        # Example 5: DuckDuckGo
        example_duckduckgo_retriever()
    except Exception as e:
        print(f"❌ DuckDuckGo error: {e}")
    
    try:
        # Example 6: Google Serper
        example_google_serper_retriever()
    except Exception as e:
        print(f"❌ Google Serper error: {e}")
    
    try:
        # Example 7: ArXiv Advanced
        example_arxiv_advanced()
    except Exception as e:
        print(f"❌ ArXiv Advanced error: {e}")
    
    # Show comparison
    print("\n" + "="*70)
    print("RETRIEVER COMPARISON")
    print("="*70)
    print(RETRIEVER_COMPARISON)
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
✅ For General Questions:
   → Use WikipediaRetriever (no setup needed)

✅ For Academic/Scientific Questions:
   → Use ArxivRetriever (papers) or PubMedRetriever (medical)

✅ For Current News/Trending Topics:
   → Use DuckDuckGoSearchAPIRetriever (no API key)
   → Or Google Serper API (premium, ~$0.05/search)

✅ For Production Systems:
   → Google Serper API (reliable, official)
   → Or build custom retriever over your own database
    """)
