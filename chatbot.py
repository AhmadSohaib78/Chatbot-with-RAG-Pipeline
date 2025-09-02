#!/usr/bin/env python3
"""
Research Papers Chatbot

A simple chatbot that answers questions based on the research papers database.
Uses the complete pipeline to search and provide relevant answers.
Supports both local models and external API services.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not available. Install with: pip install python-dotenv")

# Set hardcoded API key
client = OpenAI(api_key="YOUR_API_KEY_HERE")


# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vector_store_psycopg2 import VectorStorePsycopg2, IndexType

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
    EMBEDDING_MODEL_AVAILABLE = False

# Try to import external API libraries
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    print("Warning: anthropic not available. Install with: pip install anthropic")
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: openai not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False


class ResearchPapersChatbot:
    """Simple chatbot for research papers."""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 llm_provider: str = "openai",  # "local", "anthropic", "openai"
                 api_key: Optional[str] = None,
                 llm_model: str = "gpt-4o-mini"):
        """Initialize the chatbot."""
        self.vector_store = VectorStorePsycopg2(table_name="research_papers_complete")
        self.llm_provider = llm_provider
        self.api_key = api_key
        self.llm_model = llm_model
        
        # Initialize embedding model
        if EMBEDDING_MODEL_AVAILABLE:
            print(f"Loading embedding model: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        else:
            print("Using simulated embeddings (1536 dimensions)")
            self.embedding_model = None
            self.embedding_dimension = 1536
        
        # Initialize LLM client
        self.llm_client = self._initialize_llm_client()
        
        # Initialize vector store
        self._initialize_vector_store()
        
        # Chat history
        self.chat_history = []
    
    def _initialize_llm_client(self):
        """Initialize the LLM client based on provider."""
        if self.llm_provider == "openai" and OPENAI_AVAILABLE:
            if not self.api_key:
                self.api_key = os.environ.get("OPENAI_API_KEY")
            if self.api_key:
                print(f"Initializing OpenAI client with model: {self.llm_model}")
                openai.api_key = self.api_key
                return openai
            else:
                print("Warning: No OpenAI API key provided. Falling back to local generation.")
                return None
        
        elif self.llm_provider == "anthropic" and ANTHROPIC_AVAILABLE:
            if not self.api_key:
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
            if self.api_key:
                print(f"Initializing Anthropic client with model: {self.llm_model}")
                return anthropic.Anthropic(api_key=self.api_key)
            else:
                print("Warning: No Anthropic API key provided. Falling back to local generation.")
                return None
        
        else:
            print("Using local response generation")
            return None
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            self.vector_store.initialize(
                embedding_dimension=self.embedding_dimension,
                index_type=IndexType.HNSW,
                enable_sparse_vectors=False
            )
            print(f"Vector store initialized with {self.embedding_dimension} dimensions")
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        if self.embedding_model:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        else:
            # Simulate embedding
            import numpy as np
            np.random.seed(hash(text) % 2**32)
            return np.random.normal(0, 1, self.embedding_dimension).tolist()
    
    def search_papers(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant papers."""
        try:
            query_embedding = self._generate_embedding(query)
            
            # Perform search
            results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                metadata_filter=None,
                limit=limit
            )
            
            return results
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _generate_llm_response(self, question: str, context: str, sources: List[Dict]) -> str:
        """Generate response using external LLM or local generation."""
        if self.llm_provider == "openai" and self.llm_client:
            try:
                # Format sources for OpenAI
                sources_text = "\n\n".join([
                    f"Source {i+1}: {source['filename']} - {source['title']}"
                    for i, source in enumerate(sources)
                ])
                
                prompt = f"""You are a helpful research assistant. Answer the user's question based on the provided context from research papers.

Context from research papers:
{context}

Sources:
{sources_text}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, say so. Always cite the sources when possible."""

                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    max_tokens=1000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return self._generate_local_response(question, context, sources)
        
        elif self.llm_provider == "anthropic" and self.llm_client:
            try:
                # Format sources for Anthropic
                sources_text = "\n\n".join([
                    f"Source {i+1}: {source['filename']} - {source['title']}"
                    for i, source in enumerate(sources)
                ])
                
                prompt = f"""You are a helpful research assistant. Answer the user's question based on the provided context from research papers.

Context from research papers:
{context}

Sources:
{sources_text}

Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain enough information to answer the question, say so. Always cite the sources when possible.

Answer:"""

                response = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=1000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                return response.content[0].text
                
            except Exception as e:
                print(f"Anthropic API error: {e}")
                return self._generate_local_response(question, context, sources)
        
        else:
            return self._generate_local_response(question, context, sources)
    
    def _generate_local_response(self, question: str, context: str, sources: List[Dict]) -> str:
        """Generate response using local logic."""
        # Simple local response generation
        answer_parts = []
        
        # Main answer
        if len(context) > 500:
            # Take the most relevant parts
            sentences = context.split('.')
            relevant_sentences = sentences[:3]  # Take first 3 sentences
            main_content = '. '.join(relevant_sentences) + '.'
        else:
            main_content = context
        
        answer_parts.append(f"Based on the research papers, here's what I found:\n\n{main_content}")
        
        # Add sources
        if sources:
            answer_parts.append("\n\n**Sources:**")
            for i, source in enumerate(sources[:2], 1):  # Show top 2 sources
                answer_parts.append(f"{i}. {source['filename']} - {source['title']} (relevance: {source['similarity']:.2f})")
        
        return "\n".join(answer_parts)
    
    def generate_answer(self, question: str) -> str:
        """Generate an answer based on the question and search results."""
        # Search for relevant content
        search_results = self.search_papers(question, limit=3)
        
        if not search_results:
            return "I couldn't find any relevant information in the research papers to answer your question. Please try rephrasing your question or ask about a different topic."
        
        # Extract relevant information
        relevant_content = []
        sources = []
        
        for result in search_results:
            metadata = result.get('metadata', {})
            content = result.get('chunk', '')
            similarity = result.get('similarity', 0)
            
            if similarity > 0.3:  # Only include relevant results
                relevant_content.append(content)
                sources.append({
                    'filename': metadata.get('filename', 'Unknown'),
                    'title': metadata.get('title', 'Untitled'),
                    'similarity': similarity
                })
        
        if not relevant_content:
            return "I found some papers but they don't seem directly relevant to your question. Could you please rephrase your question?"
        
        # Combine context
        combined_context = " ".join(relevant_content)
        
        # Generate answer using LLM
        answer = self._generate_llm_response(question, combined_context, sources)
        
        # Add to chat history
        self.chat_history.append({
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'sources': sources
        })
        
        return answer
    
    def get_chat_history(self) -> List[Dict]:
        """Get chat history."""
        return self.chat_history
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics."""
        try:
            # Try to get some basic info
            return {
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_dimensions': self.embedding_dimension,
                'llm_provider': self.llm_provider,
                'llm_model': self.llm_model,
                'chat_history_length': len(self.chat_history)
            }
        except Exception as e:
            return {
                'error': str(e),
                'chat_history_length': len(self.chat_history)
            }


def main():
    """Main chatbot interface."""
    print("🤖 Research Papers Chatbot")
    print("=" * 50)
    print("Ask questions about research papers and get AI-powered answers!")
    print("Type 'quit' to exit, 'history' to see chat history, 'clear' to clear history")
    print("=" * 50)
    
    # Check for API key in environment or command line
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    
    # Initialize chatbot with OpenAI GPT-4o-mini as default
    if api_key and OPENAI_AVAILABLE:
        print("Using OpenAI GPT-4o-mini for enhanced responses")
        chatbot = ResearchPapersChatbot(
            llm_provider="openai",
            api_key=api_key,
            llm_model="gpt-4o-mini"
        )
    elif api_key and ANTHROPIC_AVAILABLE:
        print("Using Anthropic API for enhanced responses")
        chatbot = ResearchPapersChatbot(
            llm_provider="anthropic",
            api_key=api_key,
            llm_model="claude-3-haiku-20240307"
        )
    else:
        print("Using local response generation")
        chatbot = ResearchPapersChatbot(llm_provider="local")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("👋 Goodbye! Thanks for using the Research Papers Chatbot!")
                break
            
            elif user_input.lower() == 'history':
                history = chatbot.get_chat_history()
                if history:
                    print("\n📚 Chat History:")
                    for i, entry in enumerate(history[-5:], 1):  # Show last 5 entries
                        print(f"{i}. Q: {entry['question'][:50]}...")
                        print(f"   A: {entry['answer'][:100]}...")
                        print()
                else:
                    print("No chat history yet.")
                continue
            
            elif user_input.lower() == 'clear':
                chatbot.clear_history()
                print("🗑️ Chat history cleared!")
                continue
            
            elif user_input.lower() == 'stats':
                stats = chatbot.get_statistics()
                print(f"\n📊 Statistics:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            
            # Generate answer
            print("\n🤖 Bot: Thinking...")
            answer = chatbot.generate_answer(user_input)
            print(f"\n🤖 Bot: {answer}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the Research Papers Chatbot!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again or type 'quit' to exit.")


if __name__ == "__main__":
    main() 