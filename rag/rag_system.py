from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from query_processor import QueryProcessor
import torch
import os
import glob
import re

# Fixed MD document path (consistent with the OCR processor)
MD_OUTPUT_FOLDER = "./md"


class RAGSystem:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.query_processor = None

    def setup_llm(self, model_path="../llm/DeepSeek-R1-0528-Qwen3-8B"):
        """Set up the language model."""
        """Use vLLM to accelerate inference."""
        print("Loading the language model with vLLM...")

        try:
            from langchain_openai import ChatOpenAI
            # from langchain_community.llms import VLLM

            self.llm = llm = ChatOpenAI(
                model="../llm/DeepSeek-R1-0528-Qwen3-8B",
                openai_api_base="http://localhost:8000/v1",
                openai_api_key="EMPTY",
                max_tokens=2048,
                temperature=0.3,
                top_p=0.9,
            )
            # self.llm = VLLM(
            #     model=model_path,
            #     trust_remote_code=True,
            #     max_new_tokens=2048,
            #     temperature=0.3,
            #     top_p=0.9,
            #     repetition_penalty=1.1,
            #     tensor_parallel_size=1,  # Multi-GPU parallelism; adjust based on your GPU count
            #     gpu_memory_utilization=0.5,  # GPU memory usage
            #     # vLLM-specific parameters
            #     max_model_len=4096,  # Maximum model length
            #     # enable_prefix_caching=True,  # Enable prefix caching to speed up inference
            #     # max_num_seqs=16,  # Maximum number of sequences
            #     # max_num_batched_tokens=2048,  # Maximum number of batched tokens
            # )

            print("vLLM language model loaded successfully")

        except Exception as e:
            print("vLLM loading failed, falling back to the original loading method")
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=2048,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            self.llm = HuggingFacePipeline(pipeline=pipe)
        print("Language model loaded successfully")

        # Initialize the query processor
        self.setup_query_processor()

    def setup_query_processor(self):
        """Set up the query processor."""
        print("Initializing the query processor...")
        self.query_processor = QueryProcessor(self.llm)
        print("Query processor initialized successfully")

    def setup_embeddings(self, embedding_model_path="/root/.cache/modelscope/hub/models/BAAI/bge-m3"):
        """Set up the embedding model."""
        print("Loading the embedding model...")

        try:
            if os.path.exists(embedding_model_path):
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_path,
                    model_kwargs={
                        'device': 'cuda:1',  # Specify using the second GPU
                        # 'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                        'trust_remote_code': True,
                    },
                    encode_kwargs={
                        'normalize_embeddings': True,
                        'batch_size': 2,
                    }
                )
                # Test the embedding model
                test_emb = self.embeddings.embed_documents(["test"])
                if test_emb and len(test_emb[0]) > 0:
                    print("Embedding model loaded successfully")
                    return

            print("Using random embeddings")
            self.embeddings = FakeEmbeddings(size=384)

        except Exception as e:
            print(f"Embedding model failed to load: {e}")
            self.embeddings = FakeEmbeddings(size=384)

    def load_md_documents(self):
        """Load all MD documents from the fixed path."""
        print("Loading MD documents...")

        if not os.path.exists(MD_OUTPUT_FOLDER):
            print(f"MD document folder does not exist: {MD_OUTPUT_FOLDER}")
            return []

        md_files = glob.glob(os.path.join(MD_OUTPUT_FOLDER, "*.md"))
        if not md_files:
            print(f"No MD files found in '{MD_OUTPUT_FOLDER}'")
            return []

        print(f"Found {len(md_files)} MD files")
        all_documents = []

        for md_file in md_files:
            try:
                loader = TextLoader(md_file, encoding='utf-8')
                documents = loader.load()

                for doc in documents:
                    doc.metadata['source'] = os.path.basename(md_file)

                all_documents.extend(documents)
                print(f"Successfully loaded: {os.path.basename(md_file)}")

            except Exception as e:
                print(f"Failed to load {md_file}: {e}")
                continue

        return all_documents

    def retriever_vector_store(self):
        """Check whether the vector store exists."""
        if os.path.exists("./faiss"):
            try:
                from langchain_community.vectorstores import FAISS
                self.vectorstore = FAISS.load_local(
                    "./faiss",
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load FAISS; it will be recreated: {e}")

    def _is_academic_content(self, text):
        """Check whether the content is valid academic content."""
        # Filter out page numbers, headers, footers, etc.
        exclusion_patterns = [
            r'^\d+$',  # Pure numbers
            r'^[ivxlcdm]+$',  # Roman numerals
            r'^(abstract|keywords|references?)$',  # Section headings appearing as standalone blocks
            r'^.*\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec).*\d{4}.*$',  # Dates
        ]

        for pattern in exclusion_patterns:
            if re.match(pattern, text.lower().strip()):
                return False

        # Should contain enough substantive content
        words = len(text.split())
        return words >= 5  # At least 5 words

    def setup_vector_store_optimized_fallback(self, documents):
        """Optimized traditional chunking for academic papers (fallback approach)."""

        def clean_text_content(text):
            if not text or not text.strip():
                return ""
            cleaned = re.sub(r'\s+', ' ', text.strip())
            cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
            return str(cleaned) if cleaned else ""

        for doc in documents:
            doc.page_content = clean_text_content(doc.page_content)

        documents = [doc for doc in documents if doc.page_content.strip()]

        # Academic papers need larger chunk sizes
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Increase to accommodate complete paragraphs
            chunk_overlap=200,  # Increase overlap to preserve context
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "．", "；", "，", " "]
        )

        texts = text_splitter.split_documents(documents)
        processed_texts = []

        for text in texts:
            content = clean_text_content(text.page_content)
            if content and len(content.strip()) >= 50 and self._is_academic_content(content):
                text.page_content = content
                processed_texts.append(text)

        print(f"Traditional chunking complete, {len(processed_texts)} text chunks in total")

        try:
            from langchain_community.vectorstores import FAISS
            vectorstore = FAISS.from_documents(processed_texts, self.embeddings)
            vectorstore.save_local("./faiss")
            print("Vector database created successfully")
            return vectorstore
        except Exception as e:
            print(f"Vector database creation failed: {e}")
            return None

    def setup_vector_store_semantic_arxiv(self, documents):
        """Semantic chunking specifically optimized for arXiv papers."""
        if not documents:
            print("No documents to process")
            return None,[]

        # Text cleaning for academic papers
        def clean_arxiv_content(text):
            if not text or not text.strip():
                return ""

            # Preserve math formula markers
            cleaned = re.sub(r'\s+', ' ', text.strip())
            # Preserve common math and environment markers
            cleaned = re.sub(r'\\begin\{.*?\}.*?\\end\{.*?\}', lambda m: m.group(0).replace('\n', ' '), cleaned)
            cleaned = re.sub(r'\$\$.*?\$\$', lambda m: m.group(0).replace('\n', ' '), cleaned)
            cleaned = re.sub(r'\$.*?\$', lambda m: m.group(0).replace('\n', ' '), cleaned)

            # Clean control characters but preserve important markers
            cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
            return str(cleaned) if cleaned else ""

        for doc in documents:
            doc.page_content = clean_arxiv_content(doc.page_content)

        documents = [doc for doc in documents if doc.page_content.strip()]

        try:
            # Fix: use the correct SemanticChunker parameters
            text_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=85,  # Main parameter controlling chunk size
            )

            texts = text_splitter.split_documents(documents)

            # Academic-paper-specific post-processing - manually control chunk size
            processed_texts = []
            for text in texts:
                content = clean_arxiv_content(text.page_content)
                if not content:
                    continue

                # Manually filter out chunks that are too small
                if len(content.strip()) < 50:  # Academic text needs a longer minimum length
                    continue

                # Manually handle chunks that are too large - split them a second time
                if len(content) > 1200:
                    # Use a traditional splitter to subdivide chunks that are too large
                    secondary_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=150,
                        separators=["\n", "。", "！", "？", "；", "，", " "]
                    )
                    sub_chunks = secondary_splitter.split_text(content)
                    for i, sub_chunk in enumerate(sub_chunks):
                        if len(sub_chunk.strip()) >= 50 and self._is_academic_content(sub_chunk):
                            sub_doc = text.copy()
                            sub_doc.page_content = sub_chunk
                            processed_texts.append(sub_doc)
                else:
                    # Check whether it is valid academic content
                    if self._is_academic_content(content):
                        text.page_content = content
                        processed_texts.append(text)

            print(f"Semantic chunking complete, {len(processed_texts)} semantic chunks in total")

            # Analyze the chunk size distribution
            sizes = [len(t.page_content) for t in processed_texts]
            if sizes:
                avg_size = sum(sizes) / len(sizes)
                print(f"Average chunk size: {avg_size:.0f} characters")
                print(f"Min/max chunk: {min(sizes)}/{max(sizes)} characters")

            from langchain_community.vectorstores import FAISS
            vectorstore = FAISS.from_documents(processed_texts, self.embeddings)
            vectorstore.save_local("./faiss")
            print("Vector database created successfully")
            return vectorstore

        except Exception as e:
            print(f"Semantic chunking failed: {e}")
            # Fall back to the optimized traditional method
            return self.setup_vector_store_optimized_fallback(documents)

    def get_all_documents_from_faiss(self):
        """Get all document chunks from FAISS, for use by the BM25 retriever."""
        if self.vectorstore is None:
            print("FAISS vector store is not initialized")
            return []

        documents = []
        # Iterate over the FAISS docstore to get all documents
        for doc_id, doc in self.vectorstore.docstore._dict.items():
            documents.append(doc)

        print(f"Extracted {len(documents)} document chunks from FAISS")
        return documents

    def create_bm25_retriever_from_faiss(self, k=5):
        """Create a BM25 retriever based on the documents in FAISS."""
        documents = self.get_all_documents_from_faiss()
        if not documents:
            print("Unable to create the BM25 retriever: no documents")
            return None

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        try:
            bm25_retriever = BM25Retriever.from_texts(texts, metadatas=metadatas)
            bm25_retriever.k = k
            print("BM25 retriever created successfully (based on FAISS documents)")
            return bm25_retriever
        except Exception as e:
            print(f"BM25 retriever creation failed: {e}")
            return None

    def setup_hybrid_retriever(self,bm25_weight=0.4, vector_weight=0.6):
        """Set up the hybrid retriever combining BM25 and cosine similarity."""
        print("Setting up the hybrid retriever...")

        try:
            # Create the BM25 retriever
            bm25_retriever = self.create_bm25_retriever_from_faiss(k=5)

            vector_retriever = self.vectorstore.as_retriever(
                search_type="similarity",  # Use cosine similarity
                search_kwargs={"k": 5, "score_threshold": 0.7}
            )

            # 3. Create the hybrid retriever
            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[bm25_weight, vector_weight]  # Adjustable weights
            )

            print("Hybrid retriever set up successfully")
            return self.ensemble_retriever

        except Exception as e:
            print(f"Hybrid retriever setup failed: {e}")
            # Fall back to vector retrieval
            return self.setup_fallback_retriever()

    def setup_fallback_retriever(self):
        """Set up the fallback retriever."""
        return self.vectorstore.as_retriever(search_kwargs={"k": 5})

    def setup_rag_chain(self):
        """Set up the RAG chain."""
        rag_prompt_template = """Based on the multiple documents provided below, answer the user's question in Chinese. The answer should be accurate, detailed, and based on the facts in the documents.

        Document content:
        {context}

        User question: {question}

        Please answer strictly according to the following requirements:
        1. Base your answer on the provided document content; do not fabricate information
        2. If the document content is insufficient to answer the question, state this clearly
        3. The answer should be clearly structured and organized into paragraphs
        4. Cite specific information from the documents to support your answer
        5. If information comes from multiple documents, indicate the source of the information

        Answer:"""

        prompt = PromptTemplate.from_template(rag_prompt_template)

        def format_docs(docs):
            if not docs:
                return "No relevant document content found"
            formatted = []
            for i, doc in enumerate(docs, 1):
                clean_content = doc.page_content.replace('\n', ' ').strip()
                if not clean_content:
                    continue
                source = doc.metadata.get('source', 'Unknown file')
                formatted.append(f"[Document {i} - {source}]\n{clean_content}")
            return "\n\n".join(formatted) if formatted else "No relevant document content found"

        self.rag_chain = (
                {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )

    def enhanced_ask_question(self, question, show_docs=True):
        """Enhanced question-answering method that includes query decomposition and document evaluation."""
        print("Processing the complex query...")

        # 1. Query decomposition
        sub_questions = self.query_processor.decompose_query(question)
        print(f"Decomposed into {len(sub_questions)} sub-questions")

        all_relevant_docs = []
        answers = []

        # 2. Process each sub-question
        for i, sub_q in enumerate(sub_questions, 1):
            print(f"\nProcessing sub-question {i}: {sub_q}")

            # Retrieve documents
            relevant_docs = self.retriever.invoke(sub_q)
            if not relevant_docs:
                print("No relevant documents were retrieved")
                continue

            # Document relevance evaluation
            filtered_docs = []
            for doc in relevant_docs:
                is_relevant = self.query_processor.grade_document(
                    doc.page_content, sub_q
                )
                if is_relevant:
                    filtered_docs.append(doc)

            if filtered_docs:
                all_relevant_docs.extend(filtered_docs)

                # Generate a sub-answer
                context = "\n\n".join([doc.page_content for doc in filtered_docs])
                sub_answer = self.query_processor.generate_answer(context, sub_q)
                answers.append(f"Sub-question {i}: {sub_q}\nAnswer: {sub_answer}")

            if show_docs:
                print(f"Retrieved {len(filtered_docs)} relevant documents")
                for j, doc in enumerate(filtered_docs, 1):
                    source = doc.metadata.get('source', 'Unknown file')
                    content_preview = doc.page_content.replace('\n', ' ')[:120]
                    print(f"   Document {j} [{source}]: {content_preview}...")

        # 3. Synthesize all answers
        if answers:
            final_context = "\n\n".join([doc.page_content for doc in all_relevant_docs])
            combined_answer = self.query_processor.generate_answer(final_context, question)

            # Add the sub-question answers as reference
            detailed_answer = f"{combined_answer}\n\n=== Detailed analysis ===\n" + "\n\n".join(answers)
            return detailed_answer, all_relevant_docs
        else:
            # If no relevant documents were found, use the original method
            return self.ask_question(question, show_docs,use_enhanced=False)

    def ask_question(self, question, show_docs=True, use_enhanced=True):
        """Answer a question."""
        if use_enhanced and self.query_processor:
            return self.enhanced_ask_question(question, show_docs)

        # Keep the original logic unchanged
        try:
            if show_docs:
                relevant_docs = self.retriever.invoke(question)
                if relevant_docs:
                    print(f"Retrieved {len(relevant_docs)} relevant documents:")
                    for j, doc in enumerate(relevant_docs, 1):
                        source = doc.metadata.get('source', 'Unknown file')
                        content_preview = doc.page_content.replace('\n', ' ')[:120]
                        print(f"   Document {j} [{source}]: {content_preview}...")
                else:
                    print("No relevant documents were retrieved")

            answer = self.rag_chain.invoke(question)
            return answer, relevant_docs if 'relevant_docs' in locals() else []

        except Exception as e:
            print(f"An error occurred while processing the question: {e}")
            return None, None

    def initialize(self,hybrid_weights=(0.4, 0.6)):
        """Initialize the entire RAG system."""
        self.setup_llm()
        self.setup_embeddings()

        # documents = self.load_md_documents()
        # if not documents:
        #     print("No documents were loaded. The RAG system will use an empty database")
        #     return False
        self.retriever_vector_store()

        if self.vectorstore is None:
            return False

        self.retriever = self.setup_hybrid_retriever(
            bm25_weight=hybrid_weights[0],
            vector_weight=hybrid_weights[1]
        )

        self.setup_rag_chain()
        return True

    def update_rag_system(self,chunk_strategy = "semantic_arxiv",hybrid_weights=(0.4, 0.6)):
        # Process the MD files and update the FAISS database
        documents = self.load_md_documents()
        if chunk_strategy == "semantic_arxiv":
            self.vectorstore = self.setup_vector_store_semantic_arxiv(documents)
        elif chunk_strategy == "optimized":
            self.vectorstore= self.setup_vector_store_optimized_fallback(documents)
        else:
            self.vectorstore = self.setup_vector_store_semantic_arxiv(documents)  # Default
        if self.vectorstore is None:
            return False

        self.retriever = self.setup_hybrid_retriever(
            bm25_weight=hybrid_weights[0],
            vector_weight=hybrid_weights[1]
        )

        self.setup_rag_chain()


# # Create a global RAG system instance
_global_rag_system = None


def setup_rag_system():
    """Set up and return the RAG system instance."""
    global _global_rag_system
    if _global_rag_system is None:
        _global_rag_system = RAGSystem()
        success = _global_rag_system.initialize()
        if not success:
            print("RAG system initialization failed")
            return None
    return _global_rag_system



# def get_rag_system():
#     """Get the RAG system instance."""
#     return _global_rag_system
