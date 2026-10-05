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
from typing import List, Dict, Any
import re
from datetime import datetime

print("✅ Query decomposition library imports complete")

# Define the list of keywords that trigger decomposition
DECOMPOSITION_KEYWORDS = {
    "logical relation": ["and", "or", "compare", "difference", "diff", "respectively", "who is more", "both", "also", "not only", "but also", "as well as", "at the same time", "in contrast", "relative"],
    "complex intent": ["why", "how to solve", "steps", "workflow", "analyze", "summarize", "recommend", "argue", "reason", "countermeasure", "optimization plan", "how to", "how to implement", "method", "strategy", "mechanism", "principle"],
    "multi-attribute/multi-dimension": ["pros and cons", "advantages and disadvantages", "performance", "price", "features", "applicable scenarios", "deployment difficulty", "learning curve", "cost", "efficiency", "reliability", "scalability", "compatibility", "user experience", "maintenance"]
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
    entities = re.split(r'[，,、\s]+', query)
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
        print("  ✅ Clearly 'no': no trigger word + <=1 core entity")
        return "no"
    elif has_trigger_keyword and entity_count >= 2:
        print("  🔧 Clearly 'yes': trigger word + >=2 core entities")
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

# ========================================
# Part 3: Document relevance evaluation workflow
#
# Description:
# Evaluate the relevance of the retrieved documents and judge how well they match the sub-question.
# ========================================


# Define the data model
class GradeDocuments(BaseModel):
    """Multi-dimensional document relevance evaluation."""
    binary_score: str = Field(description="Basic relevance: 'yes' or 'no'")
    relevance_score: int = Field(description="Relevance score: 1-5 points")
    key_topics: List[str] = Field(description="Matched key topics")
    missing_aspects: List[str] = Field(description="Important aspects that are missing")
    confidence: float = Field(description="Evaluation confidence: 0.0-1.0")


structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Define the multi-dimensional document relevance evaluation function
def evaluate_document_relevance(question: str, document: str) -> GradeDocuments:
    """
    Evaluate the relevance of a document to the query across multiple dimensions.

    Args:
        question: The original query question
        document: The document content to be evaluated

    Returns:
        GradeDocuments: A document containing the multi-dimensional evaluation result
    """
    print(f"  📊 Evaluating document relevance...")

    # Build the evaluation prompt template
    evaluation_system_prompt = """You are a professional academic document evaluation expert. Please strictly match key topics based on the academic terminology in the document, avoiding deviations caused by synonymous substitutions (e.g. in computer networks, "routing protocol" and "path selection protocol" must be clearly distinguished).
    Please evaluate the relevance of the document to the query question from multiple dimensions:

    Evaluation dimensions:
    1. Basic relevance: does the document directly answer the question
    2. Relevance score: 1-5 points (1 point: completely irrelevant, 5 points: highly relevant and information-rich)
    3. Key topic matching: list the main concepts in the document that are related to the question
    4. Missing aspects: point out important aspects of the question that are not covered by the document
    5. Confidence: how confident you are about the evaluation result

    Scoring criteria:
    - 5 points: the document completely answers the question with rich detail and in-depth analysis
    - 4 points: the document basically answers the question and contains relevant information
    - 3 points: the document is partially relevant, but the information is insufficient
    - 2 points: the document is slightly relevant, with limited information
    - 1 point: the document is basically unrelated to the question"""

    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", evaluation_system_prompt),
        ("human", "Query question: {question}\n\nDocument to evaluate:\n{document}")
    ])

    # Create the document evaluation chain
    document_evaluator = evaluation_prompt | structured_llm_grader

    try:
        result = document_evaluator.invoke({
            "question": question,
            "document": document
        })

        print(f"  ✅ Evaluation complete - relevance score: {result.relevance_score}/5")
        print(f"  🎯 Key topics: {', '.join(result.key_topics)}")
        print(f"  ❌ Missing aspects: {', '.join(result.missing_aspects) if result.missing_aspects else 'none'}")
        print(f"  💪 Confidence: {result.confidence:.2f}")

        return result
    except Exception as e:
        print(f"  ⚠️  Evaluation error: {e}")
        # Return the default values
        return GradeDocuments(
            binary_score="no",
            relevance_score=2,
            key_topics=[],
            missing_aspects=["Evaluation function error"],
            confidence=0.5
        )

# Query rewriting function (used to improve retrieval effectiveness)
def rewrite_query_for_retrieval(original_query: str, missing_aspects: List[str]) -> str:
    """
    Rewrite the query based on the missing aspects to improve retrieval effectiveness.

    Args:
        original_query: The original query
        missing_aspects: The missing aspects found during document evaluation

    Returns:
        str: The rewritten query
    """
    print(f"  🔄 Rewriting the query to improve retrieval effectiveness...")

    if not missing_aspects:
        return original_query

    # Build the query rewriting prompt
    rewrite_prompt = f"""Rewrite the query based on the following information to improve retrieval effectiveness:

    Original query: {original_query}
    Important aspects missing from the current retrieval results: {', '.join(missing_aspects)}

    Please generate a more precise query aimed at retrieving documents that contain the missing information above. The query should:
    1. Explicitly include the missing key concepts
    2. Remain academic and accurate
    3. Not exceed 30 words
    4. Be expressed in Chinese

    Rewritten query:"""

    try:
        response = llm.invoke(rewrite_prompt)
        rewritten_query = response.content.strip()
        print(f"  ✅ Query rewriting complete")
        print(f"  📝 Original query: {original_query}")
        print(f"  🔄 Rewritten: {rewritten_query}")
        return rewritten_query
    except Exception as e:
        print(f"  ⚠️  Query rewriting error: {e}")
        return original_query

# Retrieval and evaluation loop
def retrieval_evaluation_loop(question: str, max_iterations: int = 3) -> Dict[str, Any]:
    """
    Retrieval and evaluation loop: run at most 3 retrieval-evaluation iterations.

    Args:
        question: The question to be processed
        max_iterations: Maximum number of iterations

    Returns:
        Dict: A dictionary containing the iteration results
    """
    print(f"\n🔄 Starting the retrieval and evaluation loop...")
    print(f"🎯 Target question: {question}")
    print(f"🔄 Maximum iterations: {max_iterations}")

    loop_log = {
        "question": question,
        "iterations": [],
        "success": False,
        "final_answer": None,
        "total_iterations": 0,
        "crawler_fallback": False
    }

    current_question = question

    for iteration in range(1, max_iterations + 1):
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")
        loop_log["total_iterations"] = iteration

        # 1. Simulate document retrieval. TODO: replace with an actual document retrieval function; also, you can use top-k retrieval with vector similarity filtering, or not, depending on whether we have a large enough knowledge base
        print(f"  🔍 Simulating document retrieval...")
        retrieved_doc = mock_document_retrieval(current_question, iteration)
        # print(f"📄 Retrieved document length: {len(retrieved_doc)} characters")

        # 2. Evaluate document relevance
        evaluation = evaluate_document_relevance(current_question, retrieved_doc)

        # 3. Record the iteration information
        iteration_info = {
            "iteration": iteration,
            "question_used": current_question,
            "document_preview": retrieved_doc[:100] + "...",
            "document_full": retrieved_doc,  # Save the full document for later selection
            "evaluation": evaluation.dict(),
            "timestamp": datetime.now().isoformat()
        }
        loop_log["iterations"].append(iteration_info)

        # 4. Decide the next step based on the score
        if evaluation.relevance_score >= 4:
            print(f"✅ Document quality is excellent (score: {evaluation.relevance_score}/5)")
            print(f"🎯 Accepting the current document and generating an answer...")

            # Generate the answer. TODO: replace with an actual answer generation function
            answer = generate_answer_with_context(question, retrieved_doc, evaluation)
            loop_log["success"] = True
            loop_log["final_answer"] = answer
            break

        elif evaluation.relevance_score >= 3:
            print(f"⚠️  Document quality is average (score: {evaluation.relevance_score}/5)")
            print(f"🔄 Rewriting the query for re-retrieval...")

            # Rewrite the query
            current_question = rewrite_query_for_retrieval(
                question,
                evaluation.missing_aspects
            )

            if iteration == max_iterations:
                print(f"⏰ Maximum iterations reached; selecting the best historical result")
                # Select the document with the highest score in history
                best_iteration = max(loop_log["iterations"], key=lambda x: x["evaluation"]["relevance_score"])
                best_doc = best_iteration["document_full"]  # Use the full document
                best_evaluation = GradeDocuments(**best_iteration["evaluation"])
                print(f"🏆 Selecting the document from iteration {best_iteration['iteration']} (score: {best_evaluation.relevance_score}/5)")
                # Generate the answer. TODO: replace with an actual answer generation function
                answer = generate_answer_with_context(question, best_doc, best_evaluation)
                loop_log["success"] = True
                loop_log["final_answer"] = answer

        else:  # score <= 2
            print(f"❌ Document quality is poor (score: {evaluation.relevance_score}/5)")

            if iteration == max_iterations:
                print(f"⏰ Maximum iterations reached; triggering the crawler fallback to obtain additional information")
                # Run the crawler to obtain new material
                # The crawler can add supplementary keywords here to limit the scope; the missing parts of each iteration's document can be viewed in the log.
                crawler_result = crawler_fallback_solution(question, retrieved_doc)

                # Record the crawler result as an additional iteration
                crawler_evaluation = evaluate_document_relevance(question, crawler_result)
                crawler_info = {
                    "iteration": iteration + 1,  # Mark as an additional iteration
                    "question_used": question,
                    "document_preview": crawler_result[:100] + "...",
                    "document_full": crawler_result,
                    "evaluation": crawler_evaluation.dict(),
                    "timestamp": datetime.now().isoformat(),
                    "source": "crawler"
                }
                loop_log["iterations"].append(crawler_info)
                loop_log["crawler_fallback"] = True
                loop_log["crawler_result"] = crawler_result

                # Select the highest-scoring document among all documents including the crawler result
                best_iteration = max(loop_log["iterations"], key=lambda x: x["evaluation"]["relevance_score"])
                best_doc = best_iteration["document_full"]
                best_evaluation = GradeDocuments(**best_iteration["evaluation"])
                print(f"🏆 Selecting the document from {('iteration ' + str(best_iteration['iteration'])) if best_iteration.get('source') != 'crawler' else 'the crawler'} (score: {best_evaluation.relevance_score}/5)")

                # Generate the answer based on the best document
                answer = generate_answer_with_context(question, best_doc, best_evaluation)
                loop_log["success"] = True
                loop_log["final_answer"] = answer + "\n\n⚠️ Note: the current answer combines retrieval and crawler information; it is recommended to refer to the crawler fallback for more material."
            else:
                print(f"🔄 Continuing to the next iteration...")
                # Rewrite the query and continue
                current_question = rewrite_query_for_retrieval(
                    question,
                    evaluation.missing_aspects
                )

    return loop_log


    f"""
    This part is the same as before.
    For each sub-question,
    directly call retrieval_evaluation_loop.
    When integrating, you need to manually add the operation of concatenating Q&A pairs; below there are functions for integrating Q&A pairs and the final prompt, for reference only.
    """


# Q&A generation prompt template
# Define the answer generation prompt template
template = """The following is a set of Q&A pairs (academic material) related to the target question:
{context}

Your task is to synthesize a comprehensive, academically rigorous answer based on the above Q&A pairs. Please strictly follow the following integration requirements and output standards to ensure the accuracy, completeness, and logical coherence of the answer:

## Core integration requirements (all must be followed)
1. Deduplication and filtering: identify and remove repeated statements, redundant explanations, or duplicate data points in the Q&A pairs, keeping only unique information that is valuable for answering the question, without omitting key details;
2. Logical organization: arrange the content in a clear academic logical order (e.g. "definition -> principle -> method -> result -> conclusion", "basic concept -> in-depth analysis -> practical application", or "problem background -> core solution -> limitations"), and use a hierarchical structure (e.g. "key point + sub-point") to improve readability and avoid fragmentation;
3. Academic rigor:
   - Accurately cite the core viewpoints, data, and technical terms in the Q&A pairs; do not distort, exaggerate, or simplify the original meaning, and ensure the expression is consistent with the original material;
   - Keep the definitions and usage of technical terms consistent (if the Q&A pairs have a unified standard, follow their wording; if not, clearly mark the differences, e.g. "there are two definitions of 'routing convergence' in the text: 1. XXX (from Q&A pair 2); 2. XXX (from Q&A pair 4)");
   - Avoid subjective assumptions or unfounded inferences; all conclusions must be directly supported by information in the Q&A pairs, without adding personal opinions;
4. Completeness and coherence:
   - Cover all the core dimensions needed to answer the question (e.g. if the question involves "what it is, why, and how", respond to each; if it involves technical details, include key parameters, steps, or applicable conditions);
   - Use transition words/phrases (e.g. "first", "furthermore", "on the contrary", "in summary", "further speaking") to connect different information points, ensuring logical flow and natural contextual connection;
5. Conflict handling:
   - If there are contradictory information or opposing viewpoints in the Q&A pairs:
     1. Preferentially keep information that is supported by data, logically rigorous, or consistent with mainstream academic consensus (if the Q&A pairs imply a relevant basis);
     2. If priority cannot be determined (e.g. both viewpoints lack sufficient evidence), objectively present all conflicting perspectives and mark them as "points of dispute" with a brief explanation (e.g. "there are two opposing viewpoints on this question: viewpoint 1... (from Q&A pair 3); viewpoint 2... (from Q&A pair 5). Neither viewpoint in the available information provides sufficient argumentation, so further verification is needed");
     3. It is strictly forbidden to discard conflicting information without basis or to forcibly unify the conclusions.

## Output standards
- Format: use a formal academic writing style (avoid colloquialisms and internet slang), use a hierarchical structure (e.g. "1. Key point -> (1) sub-point" or "bold key point + detailed explanation") to ensure clarity and readability;
- Completeness check: after synthesis, confirm that no key information in the Q&A pairs has been omitted (content irrelevant to the target question can be removed);
- Supplementary note: if the Q&A pairs lack information required to answer the question (e.g. core principles, key data, key steps, or applicable scenarios), clearly note at the end: "Note: the existing Q&A pairs do not cover [the specific missing aspects (e.g. 'the mathematical derivation of the algorithm', 'experimental results under different conditions', or 'practical application cases')], so the answer may not be comprehensive; supplementary material is needed for further improvement";
- No external information: do not introduce knowledge, data, or viewpoints beyond the Q&A pairs; strictly synthesize based on the given context, and do not reveal academic content that was not mentioned.

Please synthesize the answer to the following question according to the above requirements:
{question}

Integrated complete answer:
"""




# Define the function for formatting Q&A pairs
def format_qa_pair(question, answer):
    """Format a Q&A pair into string form."""
    formatted_string = ""
    formatted_string += f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

#=====================================================================================================================================


