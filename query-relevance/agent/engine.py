
from tool.tools import (
    query_transform,
    retriever,
    value_evaluator,
    web_deep_research,
    save_sub_task_result,
    report_generator,
    render_text_description
)
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Set up the LLM
# The API key is injected via the DASHSCOPE_API_KEY environment variable; never hardcode it in the code
_dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
if not _dashscope_api_key:
    raise RuntimeError(
        "DASHSCOPE_API_KEY environment variable is not set. Before starting, run: export DASHSCOPE_API_KEY=sk-xxx"
    )

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=_dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

SYSTEM_PROMPT = """
# Role: You are an autonomous decision-making deep research Agent. Your goal is to receive the user's original query, decompose it, tackle each sub-question one by one, and finally generate a report.

# Tools Capabilities
1. `query_transform`: Must be called first. Decompose the complex question into a list of sub-questions.
2. `retriever`: Basic retrieval.
3. `value_evaluator`: Must be called after retrieval. Evaluate whether the retrieved results are sufficient to answer the sub-question. Returns "YES" or "NO".
4. `web_deep_research`: May only be called when `value_evaluator` returns "NO".
5. `save_sub_task_result`: **Key step**. Whenever you finish solving a sub-question (whether through retrieval or the crawler), you must call this tool to save the result.
6. `report_generator`: Only after **all** sub-questions have been saved via `save_sub_task_result` may you call this tool to generate the final answer.

# Execution Strategy (Thinking Loop)
You must maintain your own "to-do list". After receiving the user query:

1. **Plan**: Call `query_transform` to get the list of sub-questions [Q1, Q2, Q3...].
2. **Loop (for each sub-question Q)**:
   a. **Act**: Call `retriever(Q)`.
   b. **Check**: Call `value_evaluator(Q, docs)`.
   c. **Decide**:
      - If it returns "YES": generate an answer based on the documents -> call `save_sub_task_result(Q, answer)`.
      - If it returns "NO": call `web_deep_research(Q)` -> generate an answer based on the crawler results -> call `save_sub_task_result(Q, answer)`.
3. **Finish**: Check whether all questions in the list have been run through `save_sub_task_result`. If so, call `report_generator()` to end the task.

# Constraints
- Do **not** try to answer all questions from memory in a single conversation turn; you must rely on the `save_sub_task_result` tool to record results.
- It is **strictly forbidden** to call the crawler when the evaluation passes (YES).
- It is **strictly forbidden** to call `report_generator` before all sub-questions are completed.
- When you encounter a sub-question loop, take it one step at a time; do not try to do everything in a single Function Call. Be patient.
"""

# Use the tool descriptions to build the prompt
tool_descriptions = render_text_description(tools)
system_message = SYSTEM_PROMPT + "\n\nTools available to you:\n{tool_descriptions}".format(tool_descriptions=tool_descriptions)

# Update the prompt to include the tool descriptions
#prompt_with_tools = ChatPromptTemplate.from_messages([
    #("system", system_message),
    #("user", "{input}"),
   # MessagesPlaceholder(variable_name="agent_scratchpad"), # Must be kept: stores the Agent's reasoning and tool-call history
#])

from langchain.agents import create_agent

class RagService:
    def __init__(self):
        self.agent = create_agent(
            model=llm,
            tools = [
                query_transform,
                retriever,
                value_evaluator,
                web_deep_research,
                save_sub_task_result,
                report_generator
            ] ,
            prompt=system_message,
            verbose=True,
        )

    def run(self, query: str, thread_id: str = None):
        """Support multiple sessions (state is distinguished by thread_id)."""
        #config = {"configurable": {"thread_id": thread_id}} if thread_id else {}
        return self.agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            #config=config
        )


if __name__ == "__main__":
    user_query = "How to improve retrieval quality in a RAG system?"

    agent = RagService()
    result = agent.run(user_query)


    print("\n\n>>>>>>>>>> AGENT FINAL OUTPUT <<<<<<<<<<")
