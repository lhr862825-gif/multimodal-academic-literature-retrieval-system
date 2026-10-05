

# ========================================
# Part 1: Imports and basic configuration
# ========================================

import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from typing_extensions import NotRequired
import json

# LangChain 1.0 core imports
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse, ModelCallResult
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool, BaseTool
from langchain.tools.tool_node import ToolCallRequest
from langchain.agents import AgentState
# from langchain.runtime import Runtime
from typing import Callable

# LLM model configuration (kept compatible with the original file)
from langchain_openai import ChatOpenAI

print("[OK] LangChain 1.0 library imports complete")

# Initialize the LLM model
# Load the local .env (holds DASHSCOPE_API_KEY; this file is gitignored).
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _line in _f:
            if "=" in _line and not _line.lstrip().startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

print("[OK] LLM model initialization complete")

# ========================================
# Part 2: Agent state definition
# ========================================

class CorrectiveRAGState(AgentState):
    """Extended Agent state for tracking the Corrective RAG workflow."""
    # Inherit the messages field
    last_rag_query: NotRequired[Optional[str]]       # Current RAG query
    documents_relevant: NotRequired[Optional[bool]]  # Document relevance evaluation result
    grade_reasoning: NotRequired[Optional[str]]      # Evaluation reasoning
    used_web_fallback: NotRequired[bool]             # Whether the crawler fallback was used
    subquestion_results: NotRequired[List[Dict]]     # Sub-question processing results
    current_subquestion: NotRequired[Optional[str]]  # Currently processed sub-question
    decomposition_needed: NotRequired[bool]          # Whether the question needs decomposition
    subquestions: NotRequired[List[str]]            # List of sub-questions after decomposition

print("[OK] CorrectiveRAGState definition complete")

# ========================================
# Part 3: Agent tool definitions
# ========================================

@tool
def search_documents(query: str) -> str:
    """
    Document retrieval tool

    Args:
        query: The retrieval query string

    Returns:
        str: The retrieved document content
    """
    print("[Searching] Searching documents...")

    # Simulate the retrieval process (can be replaced with a real vector retrieval in actual use)
    mock_documents = {
        "vector database": "A vector database is a database system specifically designed to store and retrieve high-dimensional vector data...",
        "RAG": "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with generative AI...",
        "LangChain": "LangChain is a framework for developing applications based on large language models..."
    }

    # Simple keyword matching retrieval
    for key, content in mock_documents.items():
        if key in query:
            print(f"[Success] Retrieval succeeded: found a relevant document ({len(content)} characters)")
            return content

    # If no relevant document is found, return an empty result
    print("[Not Found] No relevant document found")
    return ""

@tool
def crawl_web_data(query: str) -> str:
    """
    Web crawler tool

    Args:
        query: The crawler query string

    Returns:
        str: The document content obtained by the crawler
    """
    print("[Crawling] Crawling to obtain information...")

    # Simulate the crawler process (can be replaced with a real crawler in actual use)
    mock_crawler_results = {
        "vector database": "Latest information obtained through web crawling: vector database technology is developing rapidly, and many emerging vector databases such as Pinecone, Weaviate, etc. provide powerful similarity search capabilities...",
        "RAG": "Recent research shows that RAG technology made significant progress in 2024, with new methods such as multimodal RAG and Self-RAG continually emerging...",
        "LangChain": "The LangChain ecosystem continues to expand, with new versions offering more middleware options and better performance optimizations..."
    }

    # Simple keyword matching
    for key, content in mock_crawler_results.items():
        if key in query:
            print(f"[Success] Crawler succeeded: obtained supplementary information ({len(content)} characters)")
            return content

    # Return the simulated result by default
    default_result = f"The latest information about '{query}' obtained by the crawler: relevant content obtained through web crawling, including the latest developments and practical information on this topic..."
    print(f"[Success] Crawler succeeded: obtained default supplementary information ({len(default_result)} characters)")
    return default_result

print("[OK] Agent tool definitions complete")
print(f"[Tools] Available tools: [{search_documents.name}, {crawl_web_data.name}]")

# ========================================
# Part 4: Query decomposition components (keep the original logic)
# ========================================

# Define the list of keywords that trigger decomposition
DECOMPOSITION_KEYWORDS = {
    "logical relation": ["and", "or", "compare", "difference", "diff", "respectively", "who is more", "both", "also", "not only", "but also", "as well as", "at the same time", "in contrast", "relative"],
    "complex intent": ["why", "how to solve", "steps", "workflow", "analyze", "summarize", "recommend", "argue", "reason", "countermeasure", "optimization plan", "how to", "how to implement", "method", "strategy", "mechanism", "principle"],
    "multi-attribute/multi-dimension": ["pros and cons", "advantages and disadvantages", "performance", "price", "features", "applicable scenarios", "deployment difficulty", "learning curve", "cost", "efficiency", "reliability", "scalability", "compatibility", "user experience", "maintenance"]
}

def extract_entities(query: str) -> List[str]:
    """Minimalist core entity extraction method."""
    entities = re.split(r'[，,、\s]+', query)
    entities = [entity.strip() for entity in entities if len(entity.strip()) >= 2]
    return entities

def rule_based_prejudgment(query: str) -> str:
    """Rule-based prejudgment function."""
    print(f"🔍 Rule-based prejudgment: {query}")

    # Check whether the query contains a decomposition-triggering keyword
    has_trigger_keyword = False
    for category, keywords in DECOMPOSITION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query:
                has_trigger_keyword = True
                print(f"  [Trigger] Trigger keyword: '{keyword}' (category: {category})")
                break
        if has_trigger_keyword:
            break

    # Extract core entities and count them
    entities = extract_entities(query)
    entity_count = len(entities)
    print(f"  [Entities] Core entities: {entities} (count: {entity_count})")

    # Rule-based judgment
    if not has_trigger_keyword and entity_count <= 1:
        print("  ✅ Clearly 'no': no trigger word + <=1 core entity")
        return "no"
    elif has_trigger_keyword and entity_count >= 2:
        print("  🔧 Clearly 'yes': trigger word + >=2 core entities")
        return "yes"
    else:
        print("  🤔 Ambiguous case: in between")
        return "maybe"

def llm_light_classification(query: str) -> str:
    """LLM lightweight classification function."""
    print(f"[LLM Classification] LLM lightweight classification: {query}")

    try:
        classification_template = """User query: {query}
    Task: Judge whether this query needs to be split into 2 or more independent sub-questions in order to be answered completely and accurately.
    Requirements: 1. Output only "yes" or "no"; 2. No extra explanation; 3. Implicit multi-intent queries must also be judged as "yes" (e.g. "how to optimize RAG" needs to be split into "optimization direction + specific methods")."""

        classification_prompt = ChatPromptTemplate.from_template(classification_template)
        response = llm.invoke(classification_prompt.format(query=query))
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        # Normalize the output result
        if "yes" in result:
            print("  ✅ LLM classification result: yes")
            return "yes"
        elif "no" in result:
            print("  ✅ LLM classification result: no")
            return "no"
        else:
            print(f"  ⚠️ Unrecognizable LLM output: {result}, defaulting to 'no'")
            return "no"
    except Exception as e:
        print(f"  [Warning] LLM classification error: {e}, defaulting to 'no'")
        return "no"

def generate_subquestions(query: str) -> List[str]:
    """Sub-question generation function."""
    print(f"[Sub-question] Sub-question generation: {query}")

    try:
        subquestion_template = """User's original query: {query}
    Task: Split it into 2-5 independent sub-questions, satisfying:
    1. Each sub-question corresponds to only 1 information point, with no overlap;
    2. Preserve the core context of the original query (such as time, subject, scenario) without losing key constraints;
    3. The sub-questions can be used directly for retrieval (no additional information required);
    4. Do not generate redundant sub-questions (e.g. do not split "compare A and B" into "what is A" and "what is B").
    Output format: List the sub-questions numbered 1., 2., 3., etc., with no other content."""

        subquestion_prompt = ChatPromptTemplate.from_template(subquestion_template)
        response = llm.invoke(subquestion_prompt.format(query=query))
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        # Parse the sub-question list
        sub_questions = []
        lines = result.split('\n')
        for line in lines:
            line = line.strip()
            if re.match(r'^\d+\.\s*', line):
                question = re.sub(r'^\d+\.\s*', '', line).strip()
                if question:
                    sub_questions.append(question)

        print(f"  [Success] Generated {len(sub_questions)} sub-questions:")
        for i, q in enumerate(sub_questions, 1):
            print(f"    {i}. {q}")

        return sub_questions
    except Exception as e:
        print(f"  [Warning] Error generating sub-questions: {e}")
        return []

def should_decompose_query(query: str) -> tuple[bool, List[str]]:
    """Main function for judging whether a query needs decomposition."""
    print("=" * 50)
    print("[Start] Starting to judge whether the query needs decomposition")
    print(f"[Query] Original query: {query}")
    print("=" * 50)

    # Step 1: rule-based prejudgment
    rule_result = rule_based_prejudgment(query)

    if rule_result == "no":
        print("🔚 Rule-based prejudgment result: no decomposition needed")
        return False, [query]
    elif rule_result == "yes":
        print("🔜 Rule-based prejudgment result: decomposition needed")
        sub_questions = generate_subquestions(query)
        return True, sub_questions
    else:  # rule_result == "maybe"
        print("🤔 Rule-based prejudgment result: ambiguous case, entering LLM lightweight classification")
        # Step 2: LLM lightweight classification
        llm_result = llm_light_classification(query)

        if llm_result == "no":
            print("🔚 LLM lightweight classification result: no decomposition needed")
            return False, [query]
        else:  # llm_result == "yes"
            print("🔜 LLM lightweight classification result: decomposition needed")
            sub_questions = generate_subquestions(query)
            return True, sub_questions

print("[OK] Question decomposition components defined")

# ========================================
# Part 5: Document evaluation
# ========================================

class SimpleDocumentGrader:
    """Simplified document relevance grader - only judges whether the document count is 0."""

    def evaluate_documents(self, query: str, documents: List[str]) -> Dict[str, Any]:
        """
        Evaluate the retrieval results

        Args:
            query: The query string
            documents: The list of retrieved documents

        Returns:
            Dict: contains is_relevant and reasoning
        """
        doc_count = len(documents) if documents else 0

        if doc_count == 0:
            return {
                "is_relevant": False,
                "reasoning": f"0 documents retrieved; the crawler is needed to obtain more information",
                "doc_count": 0,
                "action": "use_crawler"  # Clearly indicate the next action
            }
        else:
            return {
                "is_relevant": True,
                "reasoning": f"{doc_count} documents retrieved; the content can be used directly for answering",
                "doc_count": doc_count,
                "action": "use_retrieved_docs"  # Clearly indicate the next action
            }

class DocumentGradingMiddleware(AgentMiddleware[CorrectiveRAGState]):
    """Document grading middleware - simplified document count evaluation."""

    state_schema = CorrectiveRAGState

    def __init__(self, rag_tool_name: str = "search_documents"):
        self.rag_tool_name = rag_tool_name
        self.doc_grader = SimpleDocumentGrader()
        self._pending_grading: dict | None = None

    def before_model(self, state: CorrectiveRAGState) -> dict[str, Any] | None:
        """Apply pending evaluation results to the state."""
        # Apply any pending evaluation results
        if self._pending_grading is not None:
            updates = self._pending_grading
            self._pending_grading = None  # Clear
            return updates

        return None

    def after_model(self, state: CorrectiveRAGState) -> dict[str, Any] | None:
        """Record the state after the model call."""
        return None

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Intercept tool calls to evaluate RAG output."""
        tool_name = request.tool_call.get("name", "")
        tool_args = request.tool_call.get("args", {})

        # Execute the tool first
        result = handler(request)

        # Only evaluate when it is the RAG tool
        if tool_name == self.rag_tool_name and isinstance(result, ToolMessage):
            query = tool_args.get("query", "")
            doc_content = result.content

            # Simplified evaluation: judge whether the document count is 0
            try:
                # Assume result.content is a string; split it in some way to get the document list
                # Use simple newline splitting here to simulate multiple documents
                documents = doc_content.split('\n\n') if doc_content else []
                # Filter out empty documents
                documents = [doc.strip() for doc in documents if doc.strip()]

                evaluation = self.doc_grader.evaluate_documents(query, documents)

                print(f"\n[Evaluation] Document evaluation:")
                print(f"   Query: {query}")
                print(f"   Document count: {evaluation['doc_count']}")
                print(f"   Relevance: {evaluation['is_relevant']}")
                print(f"   Reason: {evaluation['reasoning']}")

                # Store the evaluation results - will be applied in the next before_model
                self._pending_grading = {
                    "last_rag_query": query,
                    "documents_relevant": evaluation['is_relevant'],
                    "grade_reasoning": evaluation['reasoning'],
                }

            except Exception as e:
                print(f"[Warning] Evaluation failed: {e}")

        return result

class ToolFilteringMiddleware(AgentMiddleware[CorrectiveRAGState]):
    """Tool filtering middleware - Wrap-Style implementation."""

    state_schema = CorrectiveRAGState

    def __init__(self, rag_tool_name: str, web_tool_name: str):
        self.rag_tool_name = rag_tool_name
        self.web_tool_name = web_tool_name
        self._web_search_called: bool = False

    def before_model(self, state: CorrectiveRAGState) -> dict[str, Any] | None:
        """Apply web search tracking to the state."""
        if self._web_search_called:
            self._web_search_called = False
            return {"used_web_fallback": True}
        return None

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Filter tools based on the state."""

        documents_relevant = request.state.get("documents_relevant")
        used_web_fallback = request.state.get("used_web_fallback", False)

        should_enable_web = (documents_relevant is False and not used_web_fallback)

        if should_enable_web:
            filtered_tools = request.tools
            print(f"   [Unlocked] All tools enabled: {[t.name for t in filtered_tools]}")
        else:
            filtered_tools = [t for t in request.tools if t.name == self.rag_tool_name]
            print(f"   [Locked] Filtered to: {[t.name for t in filtered_tools]}")

        modified_request = request.override(tools=filtered_tools)
        # Key: pass modified_request, not the original request, to the handler!
        return handler(modified_request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage],
    ) -> ToolMessage:
        """Track when the web search tool is called."""
        tool_name = request.tool_call.get("name", "")
        result = handler(request)

        if tool_name == self.web_tool_name:
            self._web_search_called = True
            print(f"   [Tracking] Tracking: the web search tool was called")

        return result

print("[OK] Document evaluation middleware defined")

# ========================================
# Part 6: Agent initialization and configuration (revised version)
# ========================================

# Instantiate the middleware (using the simplified grader)
grading_middleware = DocumentGradingMiddleware(
    rag_tool_name="search_documents"
)

tool_filtering_middleware = ToolFilteringMiddleware(
    rag_tool_name="search_documents",
    web_tool_name="crawl_web_data"
)

# Optimized system prompt - handles the sub-question workflow while emphasizing the importance of the original query
SYSTEM_PROMPT = """You are a professional query processing assistant with the following capabilities:
1. Use the search_documents tool to retrieve relevant information from the local knowledge base
2. Use the crawl_web_data tool to obtain the latest information from the internet
3. Evaluate the quality of the retrieval results and decide whether the crawler tool is needed
4. Generate accurate and comprehensive answers based on the obtained information

Tools available to you:
- search_documents: Retrieve documents from the local knowledge base (use first)
- crawl_web_data: Obtain the latest information from the internet (use when local retrieval returns no results)

Processing workflow:
1. First get the original user query, which is in the original_query field of the state
2. Get the list of sub-questions that currently need to be processed, which is in the subquestions field of the state
3. Process each sub-question with the following steps:
   a. Use the search_documents tool to retrieve relevant information
   b. Evaluate the retrieval results:
      - If 0 documents are retrieved (content is empty), automatically use the crawl_web_data tool to obtain supplementary information
      - If there is document content, use the retrieval results directly
   c. Generate an answer for the sub-question based on the obtained information
   d. Format the answer as a Q&A pair
4. After processing all sub-questions, integrate the answers of all sub-questions based on the original user query and generate the final comprehensive answer

Important rules:
- Always remember the original user query and ensure the final answer directly responds to the user's original question
- When processing each sub-question, consider its relevance to the original query
- Do not judge whether the question needs decomposition (decomposition has already been done externally)
- Do not modify the wording of the sub-questions
- Each sub-question must go through the complete workflow: retrieval -> evaluation -> crawler if necessary -> generate answer -> format Q&A pair
- Finally, integrate the answers of all sub-questions based on the original query to generate a complete answer, ensuring contextual coherence and logical consistency

The original user query is in the original_query field of the state, and the current sub-question list is in the subquestions field of the state. Please process each sub-question in order."""

print("[OK] Agent configuration complete")

# Create the Agent
tools = [search_documents, crawl_web_data]
print(f"[Tools] Registered tools: {[t.name for t in tools]}")

# Create the Agent instance
agent = create_agent(
    model=llm,
    tools=tools,
    state_schema=CorrectiveRAGState,
    system_prompt=SYSTEM_PROMPT,
    middleware=[grading_middleware, tool_filtering_middleware]
)
print("[OK] Agent created")
print("[Config] Agent configuration:")
print(f"   Tools: [search_documents, crawl_web_data]")
print(f"   Middleware: [grading, tool_filtering]")
print(f"   State: CorrectiveRAGState")

# ========================================
# Part 7: Agent execution function
# ========================================

def execute_corrective_rag_agent(query: str, sub_questions: List[str]) -> Dict[str, Any]:
    """
    Main function for executing the Corrective RAG Agent

    Args:
        query: The user query question (used only for logging)
        sub_questions: The list of sub-questions that have already been decomposed

    Returns:
        Dict: A dictionary containing the processing result
    """
    print("🚀 Starting to execute the Corrective RAG Agent")
    print(f"❓ User query: {query}")
    print(f"📋 Received sub-question list: {len(sub_questions)}")

    # No longer judge whether decomposition is needed; process the sub-question list directly
    print(f"📋 Will process {len(sub_questions)} sub-questions")

    # Initialize the Agent state and add the original_query field
    initial_state = {
        "messages": [HumanMessage(content=f"Please process the following sub-questions in order: {sub_questions}")],
        "subquestions": sub_questions,
        "subquestion_results": [],
        "current_subquestion": None,
        "documents_relevant": None,
        "used_web_fallback": False,
        "original_query": query  # Add the original query to the state
    }

    # Execute the Agent
    try:
        result = agent.invoke(initial_state)

        print("✅ Agent execution complete")
        print(f"📊 Final state:")
        print(f"   - Processed {len(sub_questions)} sub-questions")
        print(f"   - Used crawler fallback: {result.get('used_web_fallback', False)}")
        print(f"   - Final message count: {len(result.get('messages', []))}")

        return {
            "success": True,
            "query": query,
            "subquestions": sub_questions,
            "final_result": result,
            "messages": result.get("messages", []),
            "used_web_fallback": result.get("used_web_fallback", False),
        }

    except Exception as e:
        print(f"❌ Agent execution error: {e}")
        return {
            "success": False,
            "query": query,
            "subquestions": sub_questions,
            "error": str(e),
        }

def test_decomposition_functionality():
    """Test the query decomposition functionality."""
    print("=" * 60)
    print("🧪 Test query decomposition functionality")
    print("=" * 60)

    test_queries = [
        "What is a vector database?",
        "What is the difference between vector databases and RAG technology?",
        "How to optimize the performance and deployment of LangChain?",
        "What are the pros and cons of vector databases?"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Test case {i}: {query}")
        print("-" * 40)
        need_decompose, sub_questions = should_decompose_query(query)
        print(f"   Decomposition result: {'decomposition needed' if need_decompose else 'no decomposition needed'}")
        if need_decompose:
            print(f"   Sub-question count: {len(sub_questions)}")
            for j, sq in enumerate(sub_questions, 1):
                print(f"     {j}. {sq}")

def test_agent_execution():
    """Test the Agent execution functionality."""
    print("=" * 60)
    print("🤖 Test Agent execution functionality")
    print("=" * 60)

    # Test a simple query (no decomposition needed)
    print("\n📋 Test 1: simple query")
    simple_query = "What is a vector database?"
    need_decompose, sub_questions = should_decompose_query(simple_query)
    result1 = execute_corrective_rag_agent(simple_query, sub_questions)

    # Test a complex query (decomposition needed)
    print("\n📋 Test 2: complex query")
    complex_query = "What is the difference between vector databases and RAG technology?"
    need_decompose2, sub_questions2 = should_decompose_query(complex_query)
    result2 = execute_corrective_rag_agent(complex_query, sub_questions2)

    return [result1, result2]

def main():
    """Main function - demonstrates the complete Corrective RAG Agent functionality."""
    print("🚀 Starting the Corrective RAG Agent demo")
    print("=" * 60)

    try:
        # 1. Test the query decomposition functionality
        test_decomposition_functionality()

        print("\n" + "=" * 60)

        # 2. Test the Agent execution functionality
        test_results = test_agent_execution()

        print("\n" + "=" * 60)
        print("✅ Demo complete!")
        print("📊 Test result summary:")
        for i, result in enumerate(test_results, 1):
            status = "✅ success" if result.get("success") else "❌ failure"
            print(f"   Test {i}: {status}")
            if result.get("success"):
                print(f"     - Query: {result.get('query')}")
                print(f"     - Sub-question count: {len(result.get('subquestions', []))}")
                print(f"     - Used crawler: {'yes' if result.get('used_web_fallback') else 'no'}")

    except Exception as e:
        print(f"❌ An error occurred during the demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
