import os
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



# Query rewriting function (used to improve retrieval effectiveness)
def rewrite_query_for_retrieval(original_query: str, missing_aspects: list[str]) -> str:
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

def demonstrate_query_rewrite_example():
    """
    Example function that demonstrates the query rewriting process.
    """
    print("\n=== Query rewriting example ===")

    # Example scenario: the user asks about LLM autonomous agents
    original_query = "What is the working principle of an LLM autonomous agent?"
    missing_aspects = ["specific component composition", "decision-making workflow", "application scenarios"]

    print(f"Original query: {original_query}")
    print(f"Missing aspects: {', '.join(missing_aspects)}")

    # Simulate the rewriting process
    rewritten_query = f"{original_query} including {', '.join(missing_aspects)}"
    print(f"Rewritten query: {rewritten_query}")

    print("\nThe purpose of rewriting this way is:")
    print("1. Explicitly require the missing key information (component composition, decision workflow, application scenarios)")
    print("2. Improve the recall of the retrieval system to obtain more comprehensive information")
    print("3. Avoid retrieving documents that only introduce basic concepts and lack depth")

demonstrate_query_rewrite_example()
