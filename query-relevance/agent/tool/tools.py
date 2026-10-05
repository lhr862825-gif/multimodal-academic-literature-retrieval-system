import os
import json
from typing import List, Dict

# LangChain core components
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

import sys
import os

# Add the arxiv crawler related import
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rag"))
from arxiv_crawler_integrated import ArxivCrawlerIntegrated

# Add the tool directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
tool_dir = os.path.join(current_dir, "tool")
sys.path.insert(0, tool_dir)

# Import tool modules
from checkDecomposition import QueryDecomposerAgent
from value import RelevanceGraderTool
from langchain_core.tools import render_text_description
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

# --- 1. Define a class for managing context (used to store sub-question Q&A pairs) ---
class ResearchContext:
    def __init__(self):
        self.qa_pairs = [] # Stores [{"question": "...", "answer": "..."}]
        self.original_query = ""
        self.sub_questions = [] # Stores the list of sub-questions
        self.papers = [] # Stores the list of retrieved papers

    def reset(self):
        self.qa_pairs = []
        self.original_query = ""
        self.sub_questions = [] # Reset the list of sub-questions
        self.papers = [] # Reset the list of retrieved papers

# Initialize the global context
context = ResearchContext()

# --- 2. Define the Tools ---

@tool
def query_transform(original_query: str) -> str:
    """
    Call this first. Receive the user's original query and decompose it into a list of independent sub-questions.
    """
    print(f"\n Decomposing the query: {original_query}")
    context.reset()
    context.original_query = original_query

    # Simulate the logic of queryTransform.py
    # In a real project, call queryTransform.transform(original_query) here
    #return [
    #    f"Sub-question 1: What is the core concept of {original_query}",
    #    f"Sub-question 2: Analysis of the pros and cons of {original_query}",
    #   f"Sub-question 3: The future development of {original_query}"
    #]

    # Call check_decomposition.py
    # Create a decomposer instance and call the route_query method
    decomposer = QueryDecomposerAgent(llm_client=llm)
    # Use your existing llm instance
    needs_decomposition, reason, sub_questions = decomposer.route_query(original_query)

    context.sub_questions = sub_questions # Store the list of sub-questions

    # Build the return string with hint information
    result_info = {
        "status": "success",
        "message": f"Query optimization complete; decomposed into the following {len(sub_questions)} sub-questions",
        "sub_questions": sub_questions,
        "next_step": "Process each sub-question in turn"
    }
    return json.dumps(result_info, ensure_ascii=False)

#TODO: call the retrieval tool + return JSON format
@tool
def retriever(sub_question: str) -> str:
    """
    Retrieval tool. Used to search for relevant documents in the local vector database.
    """
    print(f"\n Retrieving: {sub_question}")
    # Call retriever.py
    return f"{{Retrieved document content: a basic introduction to {sub_question}... Next step: call value_evaluator to evaluate document relevance.}}"


@tool
def value_evaluator(sub_question: str, docs: str) -> str:
    """
    Evaluation tool. Evaluate whether the retrieved documents are sufficient to answer the sub-question.
    Returns 'PASS' (no crawler needed) or 'FAIL' (crawler needed).
    """
    print(f"\n Evaluating document relevance...")

    # Create a relevance evaluation tool instance
    grader = RelevanceGraderTool()

    # Convert the docs string to a list (if needed)
    if isinstance(docs, str):
        doc_list = [docs]
    else:
        doc_list = docs if isinstance(docs, list) else [str(docs)]

    # Call the evaluation logic in value.py using the BERT strategy
    result = grader._run(query=sub_question, documents=doc_list, strategy="bert")

    # Return 'PASS' or 'FAIL' based on the evaluation result
    # If the evaluation result indicates that the context should be used, return 'PASS'; otherwise return 'FAIL'
    if result.get("action") == "use_context":
        print(f" Document evaluation result: PASS - relevance score: {result.get('score', 0)}")
        return json.dumps({
            "status": "PASS",
            "score": result.get('score', 0),
            "message": "Document relevance score is high; no need to call the crawler tool.",
            "action": "continue_without_crawl"
        }, ensure_ascii=False)
    else:
        print(f" Document evaluation result: FAIL - relevance score: {result.get('score', 0)}")
        return json.dumps({
            "status": "FAIL",
            "score": result.get('score', 0),
            "message": "Document relevance score is low; the crawler tool needs to be called to obtain more information.",
            "action": "proceed_with_crawl"
        }, ensure_ascii=False)

@tool
def web_deep_research(sub_question: str) -> str:
    """
    Crawler tool. Call it only when value_evaluator indicates that the crawler is needed to obtain more information.
    Fetch documents from the internet and extract content.
    """
    print(f"\n Running the deep crawler: {sub_question}")

    # Create a crawler instance
    crawler = ArxivCrawlerIntegrated("./paper_results")

    # Execute step by step
    # 1. Crawl papers
    papers = crawler.crawl_papers(sub_question, max_pages=3)
    # Keep the crawled papers and store them in the database when the agent finishes
    context.papers = papers # Store the list of retrieved papers; TODO: store them in the database when the agent finishes
    #=================================================
    #TODO: save operations; just copy this code
    # 2. Save to CSV
    crawler.save_to_csv(papers, "ml_papers.csv")

    # 3. Format papers
    formatted = crawler.generate_paper_list("ml_papers.csv")
    crawler.save_formatted_papers(formatted, "formatted_ml_papers.txt")

    # 4. Download papers
    success = crawler.download_papers(max_downloads=3)

    #================================================
    # TODO: retrieval operation + return JSON format



    return f"{{Crawler content: an in-depth report about {sub_question} downloaded from the internet... Please ignore the previous retrieval result and answer the sub-question based on this content.}}"

@tool
def save_sub_task_result(sub_question: str, answer: str) -> str:
    """
    [Key tool] Save tool.
    After you have generated a satisfactory answer for a sub-question, you must call this tool to save it.
    """
    print(f"\n Saving the sub-question result: {sub_question[:10]}...")
    context.qa_pairs.append({"question": sub_question, "answer": answer})
  # return f"Saved successfully. {len(context.qa_pairs)} sub-questions completed so far. {len(context.sub_questions) - len(context.qa_pairs)} sub-questions remain. The next sub-question is: {context.sub_questions[len(context.qa_pairs)]}"
    remaining_count = len(context.sub_questions) - len(context.qa_pairs)
    next_question = context.sub_questions[len(context.qa_pairs)] if len(context.qa_pairs) < len(context.sub_questions) else "All sub-questions are complete; you can summarize now."
    return json.dumps({
        "status": "success",
        "completed_count": len(context.qa_pairs),
        "remaining_count": remaining_count,
        "next_question": next_question,
        "message": f"Sub-question answer saved successfully. {len(context.qa_pairs)} sub-questions completed so far."
    }, ensure_ascii=False)
@tool
def report_generator() -> str:
    """
    Summary tool. Call it only after all sub-questions have been saved.
    It automatically reads all the Q&A pairs saved in the background and generates the final report.
    """
    print(f"\n Generating the final report...")

    # Simulate report.py
    if not context.qa_pairs:
        return json.dumps({
            "status": "error",
            "message": "Error: no saved sub-question records were found."
        }, ensure_ascii=False)

    # Build the input for the summary model
    evidence = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in context.qa_pairs])

    final_prompt = f"""
    Based on the original question: "{context.original_query}"
    and the following sub-question research results:
    {evidence}

    Generate a comprehensive answer.
    """

    final_answer = f"{final_prompt}\n\nFinal answer:\n(Here is the integrated content...)"
    return json.dumps({
        "status": "success",
        "report_title": "Final Report",
        "analysis_count": len(context.qa_pairs),
        "original_query": context.original_query,
        "final_answer": final_answer,
        "message": f"Final report generated from detailed analysis across {len(context.qa_pairs)} dimensions"
    }, ensure_ascii=False)
