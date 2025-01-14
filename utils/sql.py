from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from typing import Dict, Any
from langgraph.graph import START, StateGraph

class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    """Type definition for SQL query output."""
    query: Annotated[str, "Syntactically valid SQL query."]

class SQLdataframeEngine:
    def __init__(self, db: SQLDatabase, llm: ChatOpenAI):
        self.db = db
        self.llm = llm
        load_dotenv()

    def write_query(self, state: State) -> Dict[str, str]:
        query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
        prompt = query_prompt_template.invoke({
            "dialect": self.db.dialect,
            "top_k": 10,
            "table_info": self.db.get_table_info(),
            "input": state["question"],
        })
        structured_llm = self.llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}

    def execute_query(self, state: State) -> Dict[str, str]:
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        return {"result": execute_query_tool.invoke(state["query"])}

    def generate_answer(self, state: State) -> Dict[str, str]:
        prompt = (
            "Given the following user question, corresponding SQL query, "
            "and SQL result, answer the user question.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}'
        )
        response = self.llm.invoke(prompt)
        return {"answer": response.content}

    def build_graph(self) -> StateGraph:
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("write_query", 
                              lambda state: self.write_query(state))
        graph_builder.add_node("execute_query",
                              lambda state: self.execute_query(state))
        graph_builder.add_node("generate_answer",
                              lambda state: self.generate_answer(state))
        
        graph_builder.add_edge(START, "write_query")
        graph_builder.add_edge("write_query", "execute_query")
        graph_builder.add_edge("execute_query", "generate_answer")
        
        return graph_builder.compile()

    def get_answer(self, user_question: str) -> str:
        graph = self.build_graph()
        steps_output = []
        
        for i, step in enumerate(graph.stream({"question": user_question}, stream_mode="updates")):
            steps_output.append(step)
        return steps_output