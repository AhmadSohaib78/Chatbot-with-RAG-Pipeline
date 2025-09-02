#!/usr/bin/env python3
"""
Complete Research Papers Pipeline with DSPy Integration

This script implements a complete pipeline:
1. Parsing (PDF/DOCX extraction)
2. Data Cleaning (remove images, tables, URLs)
3. Heading-Based Chunking (intelligent section splitting)
4. DSPy Augmentation (enhance chunks with additional context)
"""

import sys
import os
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store_psycopg2 import VectorStorePsycopg2, IndexType

class SimpleInMemoryVectorStore:
    """Simple in-memory vector store for testing without PostgreSQL."""
    
    def __init__(self, table_name: str = "research_papers"):
        self.table_name = table_name
        self.chunks = []
        self.embeddings = []
        self.metadata = []
        print(f"Initialized in-memory vector store: {table_name}")
    
    def initialize(self, embedding_dimension: int, index_type=None, enable_sparse_vectors=False):
        """Initialize the in-memory store."""
        print(f"In-memory vector store initialized with {embedding_dimension} dimensions")
        return True
    
    def add_chunks(self, chunk_data: List[Dict]) -> List[str]:
        """Add chunks to the in-memory store."""
        chunk_ids = []
        for i, data in enumerate(chunk_data):
            chunk_id = f"{self.table_name}_{len(self.chunks)}_{i}"
            self.chunks.append(data['chunk'])
            self.embeddings.append(data['embedding'])
            self.metadata.append(data['metadata'])
            chunk_ids.append(chunk_id)
        
        print(f"Added {len(chunk_ids)} chunks to in-memory store")
        return chunk_ids
    
    def hybrid_search(self, query_embedding: List[float], metadata_filter: Dict = None, limit: int = 5) -> List[Dict]:
        """Perform simple similarity search."""
        if not self.embeddings:
            return []
        
        # Simple cosine similarity calculation
        import numpy as np
        
        query_vec = np.array(query_embedding)
        similarities = []
        
        for i, embedding in enumerate(self.embeddings):
            # Apply metadata filter if provided
            if metadata_filter:
                matches_filter = True
                for key, value in metadata_filter.items():
                    if self.metadata[i].get(key) != value:
                        matches_filter = False
                        break
                if not matches_filter:
                    continue
            
            # Calculate cosine similarity
            doc_vec = np.array(embedding)
            similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append((i, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        
        for i, (doc_idx, similarity) in enumerate(similarities[:limit]):
            results.append({
                'id': f"{self.table_name}_{doc_idx}",
                'chunk': self.chunks[doc_idx],
                'metadata': self.metadata[doc_idx],
                'similarity': float(similarity)
            })
        
        return results

# Try to import document processing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    print("Warning: PyPDF2 not available. Install with: pip install PyPDF2")
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    print("Warning: python-docx not available. Install with: pip install python-docx")
    DOCX_AVAILABLE = False

# Try to import DSPy
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    print("Warning: DSPy not available. Install with: pip install dspy-ai")
    DSPY_AVAILABLE = False

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_MODEL_AVAILABLE = True
except ImportError:
    print("Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
    EMBEDDING_MODEL_AVAILABLE = False


class ResearchPaperCleaner:
    """Clean research papers by removing non-text elements."""
    
    def __init__(self):
        """Initialize the cleaner with cleaning patterns."""
        self.setup_cleaning_patterns()
    
    def setup_cleaning_patterns(self):
        """Setup regex patterns for cleaning text."""
        
        # Patterns to remove or clean
        self.patterns = {
            # Remove figure and table references
            'figure_refs': [
                r'Figure\s+\d+[\.:]?\s*[A-Za-z\s,\.\-\(\)]*',
                r'Fig\.\s*\d+[\.:]?\s*[A-Za-z\s,\.\-\(\)]*',
                r'Table\s+\d+[\.:]?\s*[A-Za-z\s,\.\-\(\)]*',
                r'Tab\.\s*\d+[\.:]?\s*[A-Za-z\s,\.\-\(\)]*',
            ],
            
            # Remove URLs and email addresses
            'urls_emails': [
                r'https?://[^\s]+',      # URLs
                r'www\.[^\s]+',          # www URLs
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # Email addresses
            ],
            
            # Remove page numbers and headers/footers
            'page_elements': [
                r'^\s*\d+\s*$',          # Standalone page numbers
                r'^\s*Page\s+\d+\s*$',   # "Page X" text
                r'^\s*[A-Za-z\s]+\s+\d+\s*$',  # Header/footer patterns
            ],
            
            # Remove table content (basic patterns)
            'table_content': [
                r'^\s*[\|\+\-\s]+\s*$',  # Table borders
                r'^\s*[\|\s]+\s*$',      # Table separators
            ],
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, pattern_list in self.patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.MULTILINE | re.IGNORECASE) 
                                              for pattern in pattern_list]
    
    def clean_text(self, text: str) -> Tuple[str, Dict]:
        """Clean text by removing non-text elements."""
        if not text:
            return "", {}
        
        original_length = len(text)
        cleaned_text = text
        
        # Step 0: Remove NUL characters and other problematic characters
        cleaned_text = cleaned_text.replace('\x00', ' ')  # Remove NUL characters
        cleaned_text = cleaned_text.replace('\x01', ' ')  # Remove SOH characters
        cleaned_text = cleaned_text.replace('\x02', ' ')  # Remove STX characters
        cleaned_text = cleaned_text.replace('\x03', ' ')  # Remove ETX characters
        cleaned_text = cleaned_text.replace('\x04', ' ')  # Remove EOT characters
        cleaned_text = cleaned_text.replace('\x05', ' ')  # Remove ENQ characters
        cleaned_text = cleaned_text.replace('\x06', ' ')  # Remove ACK characters
        cleaned_text = cleaned_text.replace('\x07', ' ')  # Remove BEL characters
        cleaned_text = cleaned_text.replace('\x08', ' ')  # Remove BS characters
        cleaned_text = cleaned_text.replace('\x0b', ' ')  # Remove VT characters
        cleaned_text = cleaned_text.replace('\x0c', ' ')  # Remove FF characters
        cleaned_text = cleaned_text.replace('\x0e', ' ')  # Remove SO characters
        cleaned_text = cleaned_text.replace('\x0f', ' ')  # Remove SI characters
        
        # Remove other control characters except newlines and tabs
        import re
        cleaned_text = re.sub(r'[\x10-\x1f\x7f-\x9f]', ' ', cleaned_text)
        
        # Step 1: Remove URLs and emails
        for pattern in self.compiled_patterns['urls_emails']:
            cleaned_text = pattern.sub(' ', cleaned_text)
        
        # Step 2: Remove figure and table references
        for pattern in self.compiled_patterns['figure_refs']:
            cleaned_text = pattern.sub(' ', cleaned_text)
        
        # Step 3: Remove page elements
        for pattern in self.compiled_patterns['page_elements']:
            cleaned_text = pattern.sub(' ', cleaned_text)
        
        # Step 4: Remove table content
        for pattern in self.compiled_patterns['table_content']:
            cleaned_text = pattern.sub(' ', cleaned_text)
        
        # Step 5: Remove excessive whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        # Step 6: Remove leading/trailing whitespace
        cleaned_text = cleaned_text.strip()
        
        # Calculate cleaning statistics
        removed_chars = original_length - len(cleaned_text)
        cleaning_percentage = (removed_chars / original_length * 100) if original_length > 0 else 0
        
        return cleaned_text, {
            'original_length': original_length,
            'cleaned_length': len(cleaned_text),
            'removed_chars': removed_chars,
            'cleaning_percentage': cleaning_percentage
        }


class HeadingBasedChunker:
    """Chunk research papers based on headings and section boundaries."""
    
    def __init__(self):
        """Initialize the chunker with heading detection patterns."""
        self.setup_heading_patterns()
    
    def setup_heading_patterns(self):
        """Setup regex patterns for detecting headings."""
        
        # Heading patterns for research papers
        self.heading_patterns = {
            # Main section headings (numbered)
            'main_sections': [
                r'^\s*\d+\.\s+[A-Z][^.!?]*[.!?]?\s*$',  # 1. Introduction
                r'^\s*\d+\.\s+[A-Z][^.!?]*\s*$',       # 1. Introduction
            ],
            
            # Subsection headings (numbered)
            'subsections': [
                r'^\s*\d+\.\d+\s+[A-Z][^.!?]*[.!?]?\s*$',  # 1.1 Background
                r'^\s*\d+\.\d+\s+[A-Z][^.!?]*\s*$',       # 1.1 Background
            ],
            
            # Common research paper sections
            'research_sections': [
                r'^\s*(Abstract|Introduction|Background|Related Work|Methodology|Methods|Approach|Experimental Setup|Experiments|Results|Discussion|Conclusion|Conclusions|References|Bibliography|Appendix|Acknowledgments?)\s*$',
                r'^\s*(ABSTRACT|INTRODUCTION|BACKGROUND|RELATED WORK|METHODOLOGY|METHODS|APPROACH|EXPERIMENTAL SETUP|EXPERIMENTS|RESULTS|DISCUSSION|CONCLUSION|CONCLUSIONS|REFERENCES|BIBLIOGRAPHY|APPENDIX|ACKNOWLEDGMENTS?)\s*$',
            ],
            
            # Figure and table references (often indicate section boundaries)
            'figure_table_refs': [
                r'^\s*Figure\s+\d+[\.:]?\s*[A-Z][^.!?]*[.!?]?\s*$',
                r'^\s*Table\s+\d+[\.:]?\s*[A-Z][^.!?]*[.!?]?\s*$',
            ],
        }
        
        # Compile patterns for efficiency
        self.compiled_patterns = {}
        for category, pattern_list in self.heading_patterns.items():
            self.compiled_patterns[category] = [re.compile(pattern, re.MULTILINE | re.IGNORECASE) 
                                              for pattern in pattern_list]
    
    def detect_headings(self, text: str) -> List[Dict]:
        """Detect headings in the text and return their positions."""
        lines = text.split('\n')
        headings = []
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # Check each heading pattern
            for category, patterns in self.compiled_patterns.items():
                for pattern in patterns:
                    if pattern.match(line):
                        headings.append({
                            'line_number': line_num,
                            'text': line,
                            'category': category,
                            'level': self.get_heading_level(category)
                        })
                        break  # Found a match, move to next line
                else:
                    continue
                break
        
        return headings
    
    def get_heading_level(self, category: str) -> int:
        """Get the hierarchical level of a heading category."""
        level_map = {
            'research_sections': 1,      # Abstract, Introduction, etc.
            'main_sections': 2,          # 1. Introduction
            'subsections': 3,            # 1.1 Background
            'figure_table_refs': 3,      # Figure 1, Table 1
        }
        return level_map.get(category, 3)

    def force_split_by_word_count(self, text: str, max_chunk_size: int = 2000) -> list:
        """Split text into chunks of at most max_chunk_size words."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_chunk_size):
            chunk_words = words[i:i+max_chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'title': 'Content Section',
                'content': chunk_text,
                'start_line': 0,
                'end_line': 0,
                'level': 1,
                'word_count': len(chunk_words)
            })
        return chunks

    def extract_sections(self, text: str, min_chunk_size: int = 200, max_chunk_size: int = 2000) -> List[Dict]:
        """Extract sections based on headings."""
        lines = text.split('\n')
        headings = self.detect_headings(text)
        
        if not headings:
            # No headings found, fall back to paragraph-based chunking
            return self.fallback_chunking(text, min_chunk_size, max_chunk_size)
        
        sections = []
        
        # Start with content before first heading
        if headings[0]['line_number'] > 0:
            intro_content = '\n'.join(lines[0:headings[0]['line_number']])
            if len(intro_content.split()) >= min_chunk_size:
                sections.append({
                    'title': 'Introduction',
                    'content': intro_content,
                    'start_line': 0,
                    'end_line': headings[0]['line_number'] - 1,
                    'level': 1,
                    'word_count': len(intro_content.split())
                })
        
        # Process sections between headings
        for i, heading in enumerate(headings):
            start_line = heading['line_number']
            
            # Find end of this section (next heading or end of document)
            if i + 1 < len(headings):
                end_line = headings[i + 1]['line_number'] - 1
            else:
                end_line = len(lines) - 1
            
            # Extract section content
            section_content = '\n'.join(lines[start_line:end_line + 1])
            word_count = len(section_content.split())
            
            # Only add if it meets size requirements
            if word_count >= min_chunk_size:
                sections.append({
                    'title': heading['text'],
                    'content': section_content,
                    'start_line': start_line,
                    'end_line': end_line,
                    'level': heading['level'],
                    'word_count': word_count
                })
        
        # Split large sections
        final_sections = []
        for section in sections:
            if section['word_count'] <= max_chunk_size:
                final_sections.append(section)
            else:
                # Force split by word count
                forced_chunks = self.force_split_by_word_count(section['content'], max_chunk_size)
                for fc in forced_chunks:
                    fc['title'] = section['title']
                    fc['level'] = section['level']
                final_sections.extend(forced_chunks)
        
        return final_sections
    
    def fallback_chunking(self, text: str, min_chunk_size: int = 200, max_chunk_size: int = 2000) -> List[Dict]:
        """Fallback to paragraph-based chunking when no headings are detected."""
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = {
            'title': 'Content Section',
            'content': '',
            'start_line': 0,
            'end_line': 0,
            'level': 1,
            'word_count': 0
        }
        
        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            
            if current_chunk['word_count'] + paragraph_words > max_chunk_size and current_chunk['content']:
                chunks.append(current_chunk.copy())
                current_chunk['content'] = paragraph
                current_chunk['word_count'] = paragraph_words
            else:
                if current_chunk['content']:
                    current_chunk['content'] += '\n\n' + paragraph
                else:
                    current_chunk['content'] = paragraph
                current_chunk['word_count'] += paragraph_words
        
        if current_chunk['content']:
            chunks.append(current_chunk)
        
        # Now force split any chunk that is still too large
        final_chunks = []
        for chunk in chunks:
            if chunk['word_count'] <= max_chunk_size:
                final_chunks.append(chunk)
            else:
                forced_chunks = self.force_split_by_word_count(chunk['content'], max_chunk_size)
                final_chunks.extend(forced_chunks)
        return final_chunks


class DSPyAugmenter:
    """Use DSPy to augment research paper chunks with additional context."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """Initialize DSPy augmenter."""
        self.available = DSPY_AVAILABLE # Always set available
        if not self.available:
            print("Warning: DSPy not available. Skipping augmentation.")
            return
        
        try:
            # Configure DSPy with hardcoded API key
            import os
            os.environ["OPENAI_API_KEY"] = "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"
            dspy.settings.configure(lm="openai/gpt-3.5-turbo")
        except Exception as e:
            print(f"Warning: Could not configure DSPy: {e}")
            self.available = False
    
    def create_augmentation_signature(self):
        """Create DSPy signature for chunk augmentation."""
        if not self.available:
            return None
        
        # Create the signature directly
        class ChunkAugmenter(dspy.Signature):
            """Augment a research paper chunk with additional context and insights."""
            
            chunk_title = dspy.InputField(desc="The title of the research paper section")
            chunk_content = dspy.InputField(desc="The content of the research paper section")
            
            augmented_content = dspy.OutputField(desc="Enhanced content with additional context, key insights, and clarifications")
            key_concepts = dspy.OutputField(desc="List of key concepts and terms mentioned in this section")
            summary = dspy.OutputField(desc="Brief summary of the main points in this section")
            related_topics = dspy.OutputField(desc="Related research topics or areas that connect to this content")
        
        return ChunkAugmenter()
    
    def augment_chunk(self, chunk: Dict) -> Dict:
        """Augment a single chunk using DSPy."""
        if not self.available:
            return chunk
        
        try:
            # Create the signature
            ChunkAugmenter = self.create_augmentation_signature()
            if not ChunkAugmenter:
                return chunk
            
            # Create an instance and call it directly
            augmenter = ChunkAugmenter()
            result = augmenter(chunk_title=chunk['title'], chunk_content=chunk['content'][:2000])
            
            # Add augmentation results to chunk
            augmented_chunk = chunk.copy()
            augmented_chunk['augmented_content'] = result.augmented_content
            augmented_chunk['key_concepts'] = result.key_concepts
            augmented_chunk['summary'] = result.summary
            augmented_chunk['related_topics'] = result.related_topics
            augmented_chunk['augmented'] = True
            
            return augmented_chunk
            
        except Exception as e:
            print(f"Warning: DSPy augmentation failed: {e}")
            chunk['augmented'] = False
            return chunk
    
    def augment_chunks(self, chunks: List[Dict], max_chunks: int = 5) -> List[Dict]:
        """Augment multiple chunks using DSPy."""
        if not self.available:
            print("DSPy not available, skipping augmentation")
            return chunks
        
        print(f"Augmenting {min(len(chunks), max_chunks)} chunks with DSPy...")
        
        augmented_chunks = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            print(f"  Augmenting chunk {i+1}/{min(len(chunks), max_chunks)}: {chunk['title'][:50]}...")
            augmented_chunk = self.augment_chunk(chunk)
            augmented_chunks.append(augmented_chunk)
        
        # Add remaining chunks without augmentation
        if len(chunks) > max_chunks:
            for chunk in chunks[max_chunks:]:
                chunk['augmented'] = False
                augmented_chunks.append(chunk)
        
        return augmented_chunks


class CompletePipeline:
    """Complete pipeline for processing research papers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_vector_store: bool = False):
        """Initialize the complete pipeline."""
        self.cleaner = ResearchPaperCleaner()
        self.chunker = HeadingBasedChunker()
        self.augmenter = DSPyAugmenter()
        
        # Initialize embedding model first
        if EMBEDDING_MODEL_AVAILABLE:
            print(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dimension = self.embedding_model.get_sentence_embedding_dimension()
        else:
            print("Using simulated embeddings (1536 dimensions)")
            self.embedding_model = None
            self.embedding_dimension = 1536
        
        # Initialize vector store (optional)
        self.use_vector_store = use_vector_store
        if use_vector_store:
            try:
                self.vector_store = VectorStorePsycopg2(table_name="research_papers_complete")
                # Initialize vector store
                self._initialize_vector_store()
            except Exception as e:
                print(f"Warning: Could not initialize PostgreSQL vector store: {e}")
                print("Falling back to in-memory vector store...")
                self.vector_store = SimpleInMemoryVectorStore(table_name="research_papers_complete")
                self.use_vector_store = True  # Keep vector store enabled with in-memory
                self._initialize_vector_store()
        else:
            self.vector_store = None
            print("Vector store disabled - processing without database storage")
    
    def _initialize_vector_store(self):
        """Initialize the vector store."""
        if not self.use_vector_store:
            return
            
        print("Initializing vector store...")
        try:
            # Check if we're using the in-memory store
            if isinstance(self.vector_store, SimpleInMemoryVectorStore):
                self.vector_store.initialize(
                    embedding_dimension=self.embedding_dimension
                )
                print(f"In-memory vector store initialized with {self.embedding_dimension} dimensions")
            else:
                # PostgreSQL store
                self.vector_store.initialize(
                    embedding_dimension=self.embedding_dimension,
                    index_type=IndexType.HNSW,
                    enable_sparse_vectors=False
                )
                print(f"PostgreSQL vector store initialized with {self.embedding_dimension} dimensions")
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
            print("Continuing without database storage...")
            self.use_vector_store = False
    
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
    
    def extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file."""
        if not PDF_AVAILABLE:
            return ""
        
        try:
            content = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            return content
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return ""
    
    def extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        if not DOCX_AVAILABLE:
            return ""
        
        try:
            doc = Document(file_path)
            content = ""
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            return content
        except Exception as e:
            print(f"Error extracting text from DOCX {file_path}: {e}")
            return ""
    
    def process_research_paper(self, file_path: str, category: str) -> Dict:
        """Process a single research paper through the complete pipeline."""
        print(f"\nProcessing: {os.path.basename(file_path)}")
        
        # Step 1: Parsing
        print("  Step 1: Parsing...")
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            raw_text = self.extract_pdf_text(file_path)
        elif file_extension == '.docx':
            raw_text = self.extract_docx_text(file_path)
        else:
            return {'error': f'Unsupported file type: {file_extension}'}
        
        if not raw_text.strip():
            return {'error': 'No text extracted'}
        
        print(f"    Extracted {len(raw_text):,} characters")
        
        # Step 2: Data Cleaning
        print("  Step 2: Data Cleaning...")
        cleaned_text, cleaning_stats = self.cleaner.clean_text(raw_text)
        
        print(f"    Original: {cleaning_stats['original_length']:,} chars")
        print(f"    Cleaned: {cleaning_stats['cleaned_length']:,} chars")
        print(f"    Removed: {cleaning_stats['removed_chars']:,} chars ({cleaning_stats['cleaning_percentage']:.1f}%)")
        
        # Step 3: Heading-Based Chunking
        print("  Step 3: Heading-Based Chunking...")
        chunks = self.chunker.extract_sections(cleaned_text, min_chunk_size=200, max_chunk_size=2000)
        
        print(f"    Created {len(chunks)} chunks")
        total_words = sum(chunk['word_count'] for chunk in chunks)
        print(f"    Total words: {total_words:,}")
        
        # Step 4: DSPy Augmentation
        print("  Step 4: DSPy Augmentation...")
        augmented_chunks = self.augmenter.augment_chunks(chunks, max_chunks=3)  # Augment first 3 chunks
        
        augmented_count = sum(1 for chunk in augmented_chunks if chunk.get('augmented', False))
        print(f"    Augmented {augmented_count} chunks")
        
        # Step 5: Store in Vector Database
        print("  Step 5: Storing in Vector Database...")
        chunk_data = []
        
        for i, chunk in enumerate(augmented_chunks):
            # Use augmented content if available, otherwise use original
            content_for_embedding = chunk.get('augmented_content', chunk['content'])
            embedding = self._generate_embedding(content_for_embedding)
            
            chunk_data.append({
                'chunk': content_for_embedding,
                'embedding': embedding,
                'metadata': {
                    'filename': os.path.basename(file_path),
                    'file_path': file_path,
                    'category': category,
                    'source': 'research_paper',
                    'file_type': file_extension[1:],
                    'chunk_index': i,
                    'total_chunks': len(augmented_chunks),
                    'word_count': chunk['word_count'],
                    'title': chunk['title'],
                    'level': chunk['level'],
                    'augmented': chunk.get('augmented', False),
                    'key_concepts': chunk.get('key_concepts', ''),
                    'summary': chunk.get('summary', ''),
                    'related_topics': chunk.get('related_topics', ''),
                    'processing_timestamp': datetime.now().isoformat()
                }
            })
        
        if self.vector_store:
            chunk_ids = self.vector_store.add_chunks(chunk_data)
            print(f"    Stored {len(chunk_ids)} chunks in database")
        else:
            print("    Skipping vector store storage as it's disabled.")
        
        return {
            'filename': os.path.basename(file_path),
            'file_path': file_path,
            'category': category,
            'parsing_stats': {'extracted_chars': len(raw_text)},
            'cleaning_stats': cleaning_stats,
            'chunking_stats': {
                'total_chunks': len(chunks),
                'total_words': total_words,
                'avg_words_per_chunk': total_words / len(chunks) if chunks else 0
            },
            'augmentation_stats': {
                'total_chunks': len(augmented_chunks),
                'augmented_chunks': augmented_count
            },
            'storage_stats': {
                'stored_chunks': len(chunk_data) if self.vector_store else 0
            },
            'success': True
        }
    
    def process_parsing_papers(self, papers_folder: str = "Group2_ResearchPapers/ParsingPapers") -> List[Dict]:
        """Process all papers in the ParsingPapers folder through the complete pipeline."""
        print("=== Complete Pipeline: Processing ParsingPapers ===\n")
        
        papers_folder_path = Path(papers_folder)
        if not papers_folder_path.exists():
            print(f"Papers folder not found: {papers_folder}")
            return []
        
        results = []
        
        for file_path in papers_folder_path.iterdir():
            if file_path.is_file():
                file_extension = file_path.suffix.lower()
                if file_extension in ['.pdf', '.docx']:
                    result = self.process_research_paper(str(file_path), "ParsingPapers")
                    results.append(result)
        
        return results
    
    def search_papers(self, query: str, limit: int = 5, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search for research papers similar to the query."""
        query_embedding = self._generate_embedding(query)
        
        # Prepare metadata filter
        metadata_filter = None
        if category_filter:
            metadata_filter = {"category": category_filter}
        
        # Perform search
        if self.vector_store:
            results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                metadata_filter=metadata_filter,
                limit=limit
            )
        else:
            print("Vector store not initialized, skipping search.")
            results = []
        
        return results


def demonstrate_complete_pipeline():
    """Demonstrate the complete pipeline."""
    print("=== Complete Research Papers Pipeline with DSPy ===\n")
    
    pipeline = CompletePipeline()
    
    # Process ParsingPapers through complete pipeline
    results = pipeline.process_parsing_papers()
    
    # Show summary
    print("\n=== Pipeline Summary ===")
    total_chunks = 0
    total_augmented = 0
    
    for result in results:
        if result.get('success'):
            print(f"\n{result['filename']}:")
            print(f"  Parsing: {result['parsing_stats']['extracted_chars']:,} chars extracted")
            print(f"  Cleaning: {result['cleaning_stats']['cleaning_percentage']:.1f}% removed")
            print(f"  Chunking: {result['chunking_stats']['total_chunks']} chunks created")
            print(f"  Augmentation: {result['augmentation_stats']['augmented_chunks']} chunks augmented")
            print(f"  Storage: {result['storage_stats']['stored_chunks']} chunks stored")
            
            total_chunks += result['storage_stats']['stored_chunks']
            total_augmented += result['augmentation_stats']['augmented_chunks']
    
    print(f"\nTotal chunks processed: {total_chunks}")
    print(f"Total chunks augmented: {total_augmented}")
    
    # Demonstrate search
    print(f"\n=== Search Demonstration ===")
    queries = [
        "document parsing techniques",
        "text extraction methods",
        "layout analysis algorithms"
    ]
    
    for query in queries:
        print(f"\nSearch: '{query}'")
        results = pipeline.search_papers(query, limit=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. Similarity: {result['similarity']:.4f}")
            print(f"     Paper: {result['metadata']['filename']}")
            print(f"     Section: {result['metadata']['title']}")
            print(f"     Augmented: {result['metadata']['augmented']}")
            if result['metadata'].get('summary'):
                print(f"     Summary: {result['metadata']['summary'][:100]}...")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Research Papers Pipeline with DSPy Integration")
    parser.add_argument("--paper", type=str, help="Path to specific paper file to process")
    parser.add_argument("--category", type=str, default="research", help="Category for the paper")
    parser.add_argument("--no-vector-store", action="store_true", help="Disable vector store storage")
    args = parser.parse_args()
    
    print("Complete Research Papers Pipeline with DSPy Integration")
    print("=" * 60)
    
    # Check dependencies
    if not PDF_AVAILABLE:
        print("Note: Install PyPDF2 for PDF processing:")
        print("pip install PyPDF2")
        print()
    
    if not DOCX_AVAILABLE:
        print("Note: Install python-docx for DOCX processing:")
        print("pip install python-docx")
        print()
    
    if not DSPY_AVAILABLE:
        print("Note: Install DSPy for augmentation:")
        print("pip install dspy-ai")
        print()
    
    if not EMBEDDING_MODEL_AVAILABLE:
        print("Note: Install sentence-transformers for embeddings:")
        print("pip install sentence-transformers")
        print()
    
    try:
        pipeline = CompletePipeline(use_vector_store=not args.no_vector_store)
        
        if args.paper:
            # Process specific paper
            print(f"\n=== Processing Specific Paper ===")
            print(f"Paper: {args.paper}")
            print(f"Category: {args.category}")
            print("=" * 60)
            
            result = pipeline.process_research_paper(args.paper, args.category)
            
            if result.get('success'):
                print(f"\n✅ Processing completed successfully!")
                print(f"📄 File: {result['filename']}")
                print(f"📊 Parsing: {result['parsing_stats']['extracted_chars']:,} chars extracted")
                print(f"🧹 Cleaning: {result['cleaning_stats']['cleaning_percentage']:.1f}% removed")
                print(f"✂️  Chunking: {result['chunking_stats']['total_chunks']} chunks created")
                print(f"🤖 Augmentation: {result['augmentation_stats']['augmented_chunks']} chunks augmented")
                print(f"💾 Storage: {result['storage_stats']['stored_chunks']} chunks stored")
            else:
                print(f"❌ Processing failed")
        else:
            # Process all papers in ParsingPapers folder (default behavior)
            demonstrate_complete_pipeline()
        
        print("\n=== Complete Pipeline Finished ===")
        print("This demonstrates:")
        print("✅ Step 1: Parsing (PDF/DOCX extraction)")
        print("✅ Step 2: Data Cleaning (remove images, tables, URLs)")
        print("✅ Step 3: Heading-Based Chunking (intelligent section splitting)")
        print("✅ Step 4: DSPy Augmentation (enhance chunks with context)")
        print("✅ Vector Database Storage (with metadata)")
        print("✅ Semantic Search (with similarity scores)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 