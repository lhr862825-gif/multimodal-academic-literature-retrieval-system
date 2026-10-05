# query_processor.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser



class GradeDocuments(BaseModel):
    """Document relevance evaluation model."""
    binary_score: str = Field(
        description="Whether the document is relevant to the question, 'yes' or 'no'"
    )


class QueryProcessor:
    def __init__(self, llm):
        self.llm = llm
        self.setup_components()

    def setup_components(self):
        """Initialize all processing components."""
        # 1. Document relevance grader
        self.setup_document_grader()

        # 2. Query decomposer
        self.setup_query_decomposer()

        # 3. Answer generator
        self.setup_answer_generator()

    def setup_document_grader(self):
        """Set up the document relevance evaluation chain."""
        system_prompt = """You are a grader that evaluates the relevance of a retrieved document to a user's question.
        If the document contains keywords or semantic meaning related to the question, grade it as relevant.
        Give a binary score of 'yes' or 'no' to indicate whether the document is relevant to the question."""

        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Retrieved document:\n\n{document}\n\nUser question: {question}"),
        ])

        # Use a local model for structured output (requires adaptation)
        self.retrieval_grader = grade_prompt | self.llm | StrOutputParser()

    def setup_query_decomposer(self):
        """Set up the query decomposition chain."""
        decomposition_template = """You are a helpful assistant that can generate multiple sub-questions related to an input question.
        The goal is to decompose the input into a set of sub-questions/sub-queries that can be answered independently.
        Generate multiple search queries related to the following: {question}
        Output (3 queries):"""

        prompt_decomposition = ChatPromptTemplate.from_template(decomposition_template)

        self.query_decomposer = (
                prompt_decomposition
                | self.llm
                | StrOutputParser()
                | (lambda x: [q.strip() for q in x.split("\n") if q.strip()][:3])
        )

    def setup_answer_generator(self):
        """Set up the answer generation chain."""
        self.answer_prompt = ChatPromptTemplate.from_template("""
Answer the question based on the following context. If the context is insufficient to answer the question, state which information is missing.

Context:
{context}

Question: {question}

Please provide an accurate and detailed answer:""")

        self.answer_chain = self.answer_prompt | self.llm | StrOutputParser()

    def decompose_query(self, question):
        """Decompose a complex query into sub-questions."""
        try:
            return self.query_decomposer.invoke({"question": question})
        except Exception as e:
            print(f"Query decomposition failed: {e}")
            return [question]  # Return the original question on failure

    def grade_document(self, document, question):
        """Evaluate document relevance."""
        try:
            response = self.retrieval_grader.invoke({
                "document": document,
                "question": question
            })
            # Parse the response to judge relevance
            return "yes" in response.lower()
        except Exception as e:
            print(f"Document evaluation failed: {e}")
            return True  # Default to relevant on failure

    def generate_answer(self, context, question):
        """Generate the final answer."""
        return self.answer_chain.invoke({
            "context": context,
            "question": question
        })
