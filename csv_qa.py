import streamlit as st
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv
import os
import io
import base64
from PIL import Image
import pandas as pd
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from lida import Manager, llm as lida_llm, TextGenerationConfig
from utils.sql import SQLdataframeEngine
from utils.python import DataFrameQueryEngine

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Agent CSV Analysis",
    page_icon="🔍",
)

if os.getenv("OPENAI_API_KEY") is None:
    api_key = st.sidebar.text_input("OpenAI API Key", "")
    os.environ["OPENAI_API_KEY"] = api_key

if os.getenv("OPENAI_API_KEY") is None:
    st.error("Please provide an OpenAI API Key to use the application.")
    st.stop()

# Initialize LIDA
@st.cache_resource
def init_lida():
    text_gen = lida_llm("openai")
    textgen_config = TextGenerationConfig(n=1, temperature=0.0, model="gpt-4o-mini", use_cache=True)
    lida = Manager(text_gen=text_gen)
    return lida, textgen_config

# SQL Database functions
@st.cache_resource
def init_database(connection_string: str = "sqlite:///db/hcr.db") -> SQLDatabase:
    return SQLDatabase.from_uri(connection_string)

@st.cache_resource
def init_llm(model_name: str = "gpt-4o-mini") -> ChatOpenAI:
    return ChatOpenAI(model=model_name)

@st.cache_data
def get_sql_answer(user_question: str, _engine) -> str:
    graph = _engine.build_graph()
    steps_output = []
    for i, step in enumerate(graph.stream({"question": user_question}, stream_mode="updates")):
        steps_output.append(step)  
    return steps_output

# Python agent functions
@st.cache_data
def get_python_response(_query_engine, input_query) -> tuple:
    try:
        return _query_engine.query(input_query)
    except Exception as e:
        raise

# LIDA visualization functions
@st.cache_data
def get_lida_chart(query, _lida, _summary, _textgen_config) -> str:
    return _lida.visualize(summary=_summary, goal=query, textgen_config=_textgen_config, library="plotly")

def display_chart(query, selected_viz):
    if selected_viz.raster:
        imgdata = base64.b64decode(selected_viz.raster)
        img = Image.open(io.BytesIO(imgdata))
        st.image(img, caption=query, use_container_width=True)

def main():
    st.title("Multi-Agent CSV Analysis")
    
    # Initialize components
    db = init_database()
    llm = init_llm()
    sql_engine = SQLdataframeEngine(db, llm)
    df = pd.read_csv("data/HCRDatabaseAnalysisStream.csv")
    python_engine = DataFrameQueryEngine(df)
    lida, textgen_config = init_lida()
    lida_summary = lida.summarize("data/HCRDatabaseAnalysisStream.csv")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["SQL Agent", "Python Agent", "Visualization"])

    # User input
    user_query = st.text_input(
        "Enter your question",
        "What is the distribution of releases by Validated_immediate_cause?"
    )

    if user_query:
        # SQL Agent Tab
        with tab1:
            st.subheader("SQL Analysis")
            steps_output = get_sql_answer(user_query, sql_engine)
            with st.expander("View Generated SQL Code", expanded=False):
                if steps_output:
                    st.code(steps_output[0]['write_query']['query'])
            if steps_output:
                final_state = steps_output[-1]
                if "generate_answer" in final_state:
                    st.markdown(final_state["generate_answer"]["answer"])
                else:
                    st.error("No answer was generated. Please try rephrasing your question.")

        # Python Agent Tab
        with tab2:
            st.subheader("Python Analysis")
            try:
                response, python_code = get_python_response(python_engine, user_query)
                with st.expander("View Generated Python Code", expanded=False):
                    st.code(python_code)
                st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred during query processing: {str(e)}")

        # LIDA Visualization Tab
        with tab3:
            st.subheader("Data Visualization")
            try:
                charts = get_lida_chart(user_query, lida, lida_summary, textgen_config)
                if charts:
                    selected_viz = charts[0]
                    display_chart(user_query, selected_viz)
                else:
                    st.warning("No visualization could be generated for this query.")
            except Exception as e:
                st.error(f"An error occurred during visualization: {str(e)}")
    else:
        st.warning("Please enter a question to analyze the data.")

if __name__ == "__main__":
    main()