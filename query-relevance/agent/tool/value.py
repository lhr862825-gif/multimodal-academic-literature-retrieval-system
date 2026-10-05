import logging
import time
from typing import List, Dict, Union, Literal
from pydantic import BaseModel, Field
import re

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RelevanceGraderInput(BaseModel):
    """Input parameter schema for the document relevance evaluation tool."""

    query: str = Field(
        ...,
        description="The user's original query question, e.g.: 'What is a vector database?'"
    )
    documents: List[str] = Field(
        ...,
        description="The list of retrieved document contents, where each element is the full text of one document"
    )
    strategy: Literal["llm", "bert"] = Field(
        "llm",
        description="Evaluation strategy: 'llm' uses a large language model for evaluation, 'bert' uses a BERT model to compute cosine similarity"
    )


class RelevanceGraderTool(BaseTool):
    """
    Document relevance evaluation tool

    This tool evaluates the relevance of retrieved documents to the user's query, helping the AI Agent decide
    whether to use the existing document context or call the crawler to obtain more information.

    Features:
    - Supports two evaluation strategies: LLM evaluation and BERT semantic similarity evaluation
    - Provides synchronous and asynchronous invocation methods
    - Automatically handles empty documents and exceptional situations
    - Returns a structured evaluation result

    Usage scenarios:
    - Judging the quality of retrieval results in a RAG system
    - Deciding whether supplementary information is needed in an intelligent Q&A system
    - Quality filtering after document retrieval
    """

    name: str = "document_relevance_grader"
    description: str = """Evaluate the relevance of retrieved documents to the question, helping decide whether more information is needed.

This tool receives the user query and the list of retrieved documents, and uses the specified strategy to evaluate document relevance:
- If the documents are relevant and contain sufficient information, it returns 'use_context', suggesting the use of the existing documents
- If the documents are irrelevant or the information is insufficient, it returns 'call_crawler', suggesting calling the crawler to obtain more information

Two evaluation strategies are supported:
1. 'llm': Uses a large language model for deep semantic understanding evaluation (high precision, slower)
2. 'bert': Uses a BERT model to compute cosine similarity (fast, suitable for real-time scenarios)

The returned result includes:
- action: 'use_context' or 'call_crawler'
- score: relevance score between 0 and 1
- reason: explanation of the decision reason
- doc_count: number of evaluated documents
"""
    args_schema: type[BaseModel] = RelevanceGraderInput

    # Internal components
    llm: ChatOpenAI = None
    embedding_model: HuggingFaceEmbeddings = None
    similarity_threshold: float = 0.75

    def __init__(self, **kwargs):
        """
        Initialize the document relevance evaluation tool

        Args:
            **kwargs: Extra arguments passed to the parent class
        """
        super().__init__(**kwargs)
        # Initialize the LLM (for the LLM evaluation strategy)
        # self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Initialize the Embedding (for the BERT evaluation strategy; loaded lazily to save resources)
        # self.embedding_model = HuggingFaceEmbeddings(model_name="m3e-base")

    def _run(self, query: str, documents: List[str], strategy: str = "bert") -> Dict:
        """
        Run document relevance evaluation synchronously

        Args:
            query: The user's original query question
            documents: The list of retrieved document contents
            strategy: Evaluation strategy, 'llm' or 'bert'

        Returns:
            Dict: The evaluation result dictionary, including the following fields:
                - action: 'use_context' or 'call_crawler'
                - score: relevance score (0-1)
                - reason: decision reason
                - latency: evaluation elapsed time (seconds)
                - doc_count: number of evaluated documents

        Example:
            >>> grader = RelevanceGraderTool()
            >>> result = grader._run(
            ...     query="What is a vector database?",
            ...     documents=["A vector database is a database system specifically designed to store and retrieve vector data..."],
            ...     strategy="bert"
            ... )
            >>> print(result['action'])
            'use_context'
        """
        start_time = time.time()
        logger.info(f"Starting evaluation, strategy: {strategy}, document count: {len(documents)}")

        # 1. Check for empty data (Pre-check)
        if not documents:
            logger.warning("The retrieval results are empty; triggering the crawler directly.")
            return {"action": "call_crawler", "reason": "empty_results", "score": 0.0}

        # 2. Run the evaluation
        try:
            if strategy == "llm":
                result = self._evaluate_with_llm(query, documents)
            elif strategy == "bert":
                result = self._evaluate_with_bert(query, documents)
            else:
                raise ValueError(f"Unsupported strategy: {strategy}")

            elapsed = time.time() - start_time
            logger.info(f"Evaluation complete, elapsed: {elapsed:.4f}s, result: {result['action']}")

            return {
                **result,
                "latency": elapsed,
                "doc_count": len(documents)
            }

        except Exception as e:
            logger.error(f"An exception occurred during evaluation: {str(e)}")
            # Fallback strategy when an exception occurs: for safety, trigger the crawler or return an error
            return {"action": "call_crawler", "reason": f"error: {str(e)}", "score": 0.0}

    async def _arun(self, query: str, documents: List[str], strategy: str = "bert") -> Dict:
        """
        Run document relevance evaluation asynchronously

        Args:
            query: The user's original query question
            documents: The list of retrieved document contents
            strategy: Evaluation strategy, 'llm' or 'bert'

        Returns:
            Dict: The evaluation result dictionary, in the same format as the _run method

        Note:
            The current implementation is a wrapper around the synchronous method; it can be optimized into a truly asynchronous implementation in the future
        """
        return self._run(query, documents, strategy)

    def _evaluate_with_llm(self, query: str, docs: List[str]) -> Dict:
        """
        Use a large language model to evaluate document relevance

        This method builds a detailed prompt so that the LLM understands the query and document content,
        and then judges whether the documents contain the key information needed to answer the query.

        Args:
            query: The user's original query question
            docs: The list of retrieved documents

        Returns:
            Dict: The evaluation result dictionary containing action, score, and reason

        Note:
            self.llm must be initialized before this method can be used
        """
        # Join the document contents; they can be truncated to save tokens
        context_text = "\n\n".join([f"Doc {i+1}: {doc[:500]}..." for i, doc in enumerate(docs)])

        prompt = ChatPromptTemplate.from_template(
            """You are a strict document relevance scorer.

            User question: {query}

            Retrieved document set:
            {context}

            Please evaluate whether the above document set contains the key information needed to answer the user's question.
            - If at least one document contains the core answer, treat it as "yes".
            - If all documents are off-topic or cannot answer the question, treat it as "no".

            Please return the result in JSON format only, in the following format:
            {{
                "score": <a float score between 0 and 1>,
                "reason": "<short reason>",
                "decision": "<'yes' or 'no'>"
            }}
            """
        )

        chain = prompt | self.llm | JsonOutputParser()
        response = chain.invoke({"query": query, "context": context_text})

        if response.get("decision") == "yes":
            return {"action": "use_context", "score": response.get("score"), "reason": response.get("reason")}
        else:
            return {"action": "call_crawler", "score": response.get("score"), "reason": response.get("reason")}

    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text and extract key information

        Args:
            text: The input text

        Returns:
            The preprocessed text
        """
        # Remove excess whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        # If the text is too long, consider keeping only the key parts
        # No truncation for now; adjust as needed

        return text

    def _evaluate_with_bert(self, query: str, docs: List[str]) -> Dict:
        """
        Use a BERT model and cosine similarity to evaluate document relevance

        This method evaluates relevance through the following steps:
        1. Convert the query and documents into vector representations
        2. Compute the cosine similarity between the query vector and each document vector
        3. Take the maximum similarity as the final score
        4. Compare with the threshold to decide whether to use the existing documents

        Args:
            query: The user's original query question
            docs: The list of retrieved documents

        Returns:
            Dict: The evaluation result dictionary containing action, score, and reason

        Note:
            - Uses the sentence-transformers/all-MiniLM-L6-v2 model
            - The similarity threshold defaults to 0.75
            - The model is loaded on the first call and reused afterwards
        """
        # Lazily load the embedding model
        if not self.embedding_model:
            logger.info("Loading the embedding model...")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

        if docs is None or len(docs) == 0:
            return {"action": "call_crawler", "score": 0.0, "reason": "no_docs"}

        # Preprocess the query and documents
        processed_query = self._preprocess_text(query)
        processed_docs = [self._preprocess_text(doc) for doc in docs]

        # Compute the query vector
        query_emb = self.embedding_model.embed_query(processed_query)

        # Compute the document vectors
        doc_embs = self.embedding_model.embed_documents(processed_docs)

        # Compute the similarity matrix
        scores = cosine_similarity([query_emb], doc_embs)[0]
        max_score = float(np.max(scores))

        logger.info(f"BERT maximum similarity score: {max_score}")

        # Improved threshold judgment logic
        # If the similarity is high, use the context directly
        if max_score >= self.similarity_threshold:
            return {
                "action": "use_context",
                "score": max_score,
                "reason": "semantic_similarity_pass"
            }
        # If the similarity is lower but above 0.6, consider further judgment
        elif max_score >= 0.6:
            # Check whether there is some semantic relevance
            avg_score = float(np.mean(scores))
            # If the average score is also high, the documents as a whole have some relevance to the query
            if avg_score >= 0.5:
                return {
                    "action": "use_context",
                    "score": max_score,
                    "reason": "moderate_similarity_with_context"
                }
            else:
                return {
                    "action": "call_crawler",
                    "score": max_score,
                    "reason": "low_similarity"
                }
        else:
            return {
                "action": "call_crawler",
                "score": max_score,
                "reason": "low_similarity"
            }


# Usage example
if __name__ == "__main__":
    # Test code
    pass
