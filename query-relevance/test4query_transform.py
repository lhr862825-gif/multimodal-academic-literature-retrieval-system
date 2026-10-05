# ========================================
# RAG query transformation and evaluation demo
#
# This script demonstrates the core components of a RAG system:
# 1. Document relevance evaluation - uses an LLM to evaluate the relevance of retrieved documents to the query
# 2. Query decomposition - decomposes a complex question into multiple sub-questions
# 3. RAG processing chain - builds a complete retrieval-generation workflow
#
# System requirements:
# - LangChain framework
# - OpenAI API access
# - An appropriate vector database and retriever
# ========================================

# ========================================
# Part 1: Document relevance evaluation component
#
# Description:
# Build an evaluation chain used to judge the relevance of retrieved documents to the user query.
# ========================================

# Import the necessary libraries
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

print("✅ Basic library imports complete")

# Define the data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

print("✅ Data model defined")
print(f"📋 Model fields: {list(GradeDocuments.__fields__.keys())}")

# Initialize the LLM model for relevance evaluation
# TODO: switch to the model we use
from langchain_openai import ChatOpenAI
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


print("✅ Relevance evaluation chain built")
print("🔗 Chain structure: prompt template -> structured output model")

# ========================================
# Part 2: Query decomposition component
#
# Description:
# Decompose a complex user question into multiple sub-questions that can be answered independently, improving retrieval efficiency.
# ========================================

# ========================================
# 2.1 Basic query decomposition
# ========================================

# Import query decomposition related libraries
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from typing import List
import re

print("✅ Query decomposition library imports complete")

# Define the list of keywords that trigger decomposition
DECOMPOSITION_KEYWORDS = {
    "logical relation": ["and", "or", "compare", "difference", "diff", "respectively", "who is more", "both", "also", "not only", "but also"],
    "complex intent": ["why", "how to solve", "steps", "workflow", "analyze", "summarize", "recommend", "argue", "reason", "countermeasure", "optimization plan", "what to do", "how to deal with"],
    "multi-attribute/multi-dimension": ["pros and cons", "advantages and disadvantages", "performance", "price", "features", "applicable scenarios", "deployment difficulty", "learning curve"]
}

print("✅ Decomposition keyword list defined")

# ========================================
# 2.2 Rule-based prejudgment
# ========================================

def extract_entities(query: str) -> List[str]:
    """
    Minimalist core entity extraction method.
    In actual use, spaCy tokenization or LLM extraction can be used.
    """
    # Use a simple tokenization method as an example here
    # In actual use, replace with a more complex entity extraction method
    entities = re.split(r'[，,、？\s]+', query)
    # Filter out empty strings and strings shorter than 2 characters
    entities = [entity.strip() for entity in entities if len(entity.strip()) >= 2]
    return entities

def rule_based_prejudgment(query: str) -> str:
    """
    Rule-based prejudgment function.
    Return value: "yes" (decomposition needed), "no" (no decomposition needed), "maybe" (ambiguous case)
    """
    print(f"🔍 Rule-based prejudgment: {query}")

    # Check whether the query contains a decomposition-triggering keyword
    has_trigger_keyword = False
    for category, keywords in DECOMPOSITION_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query:
                has_trigger_keyword = True
                print(f"  🎯 Trigger keyword: '{keyword}' (category: {category})")
                break
        if has_trigger_keyword:
            break

    # Extract core entities and count them
    entities = extract_entities(query)
    entity_count = len(entities)
    print(f"  📦 Core entities: {entities} (count: {entity_count})")

    # Rule-based judgment
    if not has_trigger_keyword and entity_count <= 1:
        print("  ✅ Clearly 'no': no trigger word and <=1 core entity")
        return "no"
    elif has_trigger_keyword and entity_count >= 2:
        print("  🔧 Clearly 'yes': trigger word and >=2 core entities")
        return "yes"
    else:
        print("  🤔 Ambiguous case: in between")
        return "maybe"

print("✅ Rule-based prejudgment function defined")

# ========================================
# 2.3 LLM lightweight classification
# ========================================

# Define the LLM lightweight classification prompt template
classification_template = """User query: {query}
Task: Judge whether this query needs to be split into 2 or more independent sub-questions in order to be answered completely and accurately.
Requirements: 1. Output only "yes" or "no"; 2. No extra explanation; 3. Implicit multi-intent queries must also be judged as "yes" (e.g. "how to optimize RAG" needs to be split into "optimization direction + specific methods")."""

classification_prompt = ChatPromptTemplate.from_template(classification_template)

def llm_light_classification(query: str) -> str:
    """
    LLM lightweight classification function.
    Quickly classify queries that cannot be determined by rules using a minimal prompt.
    """
    print(f"🧠 LLM lightweight classification: {query}")

    try:
        # Use the configured LLM model
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
            # Default handling
            print(f"  ⚠️ Unrecognizable LLM output: {result}, defaulting to 'no'")
            return "no"
    except Exception as e:
        print(f"  ⚠️ LLM classification error: {e}, defaulting to 'no'")
        return "no"

print("✅ LLM lightweight classification function defined")

# ========================================
# 2.4 Sub-question generation
# ========================================

# Define the sub-question generation prompt template
subquestion_template = """User's original query: {query}
Task: Split it into 2-5 independent sub-questions, satisfying:
1. Each sub-question corresponds to only 1 information point, with no overlap;
2. Preserve the core context of the original query (such as time, subject, scenario) without losing key constraints;
3. The sub-questions can be used directly for retrieval (no additional information required);
4. Do not generate redundant sub-questions (e.g. do not split "compare A and B" into "what is A" and "what is B").
Output format: List the sub-questions numbered 1., 2., 3., etc., with no other content."""

subquestion_prompt = ChatPromptTemplate.from_template(subquestion_template)

def generate_subquestions(query: str) -> List[str]:
    """
    Sub-question generation function.
    """
    print(f"🧩 Sub-question generation: {query}")

    try:
        # Use the configured LLM model
        response = llm.invoke(subquestion_prompt.format(query=query))
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()

        # Parse the sub-question list
        sub_questions = []
        lines = result.split('\n')
        for line in lines:
            line = line.strip()
            # Match numbered sub-questions (e.g. "1. What is an LLM?")
            if re.match(r'^\d+\.\s*', line):
                # Remove the number prefix
                question = re.sub(r'^\d+\.\s*', '', line).strip()
                if question:
                    sub_questions.append(question)

        print(f"  ✅ Generated {len(sub_questions)} sub-questions:")
        for i, q in enumerate(sub_questions, 1):
            print(f"    {i}. {q}")

        return sub_questions
    except Exception as e:
        print(f"  ⚠️ Error generating sub-questions: {e}")
        return []

print("✅ Sub-question generation function defined")

# ========================================
# 2.5 Function for judging whether a query needs decomposition
# ========================================

def should_decompose_query(query: str) -> tuple[bool, List[str]]:
    """
    Main function for judging whether a query needs decomposition.
    Return value: (whether decomposition is needed, list of sub-questions)
    """
    print("=" * 50)
    print("🔍 Starting to judge whether the query needs decomposition")
    print(f"📝 Original query: {query}")
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

print("✅ Main query decomposition judgment function defined")

# ========================================
# 2.6 Test query decomposition functionality
# ========================================

# Test the query decomposition functionality
test_questions = [
    "What is a vector database",
    "Solutions for large model hallucination",
    "What to do when RAG retrieval accuracy is low",
    "Compare the pros and cons of LangChain and LlamaIndex; which one is recommended for beginners?",
    "What were the revenues of company A and company B in 2024? Which one is higher?"
]

print("\n" + "=" * 60)
print("🧪 Test query decomposition functionality")
print("=" * 60)

for i, test_query in enumerate(test_questions, 1):
    print(f"\n--- Test case {i} ---")
    need_decompose, sub_questions = should_decompose_query(test_query)
    print(f"📋 Final result: {'decomposition needed' if need_decompose else 'no decomposition needed'}")
    if need_decompose and sub_questions:
        print("📄 Sub-questions after decomposition:")
        for j, sq in enumerate(sub_questions, 1):
            print(f"  {j}. {sq}")
    print("-" * 30)
