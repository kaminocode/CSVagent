import pandas as pd
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate
from llama_index.core.query_pipeline import QueryPipeline, Link, InputComponent

class DataFrameQueryEngine:
    """A class to handle natural language queries on pandas DataFrames."""

    INSTRUCTION_TEMPLATE = """
    1. Convert the query to executable Python code using Pandas.
    2. The final line of code should be a Python expression that can be called with the `eval()` function.
    3. The code should represent a solution to the query.
    4. PRINT ONLY THE EXPRESSION.
    5. Do not quote the expression.
    """

    PANDAS_PROMPT_TEMPLATE = """
    You are working with a pandas dataframe in Python.
    The name of the dataframe is `df`.
    This is the result of `print(df.head())`:
    {df_str}

    Follow these instructions:
    {instruction_str}
    Query: {query_str}

    Expression:
    """

    RESPONSE_SYNTHESIS_TEMPLATE = """
    Given an input question, synthesize a markdown response from the query results.
    Query: {query_str}

    Pandas Instructions (optional):
    {pandas_instructions}

    Pandas Output: {pandas_output}

    Response: 
    """

    def __init__(
        self, 
        dataframe: pd.DataFrame,
        model_name: str = "gpt-4o",
        verbose: bool = False
    ):
        self.df = dataframe
        self.llm = OpenAI(model=model_name)
        self.verbose = verbose
        self._setup_pipeline()

    def _setup_pipeline(self) -> None:
        """Set up the query processing pipeline."""
        pandas_prompt = PromptTemplate(self.PANDAS_PROMPT_TEMPLATE).partial_format(
            instruction_str=self.INSTRUCTION_TEMPLATE,
            df_str=self.df.head(5)
        )
        response_synthesis_prompt = PromptTemplate(self.RESPONSE_SYNTHESIS_TEMPLATE)

        pandas_output_parser = PandasInstructionParser(self.df)

        self.pipeline = QueryPipeline(
            modules={
                "input": InputComponent(),
                "pandas_prompt": pandas_prompt,
                "llm1": self.llm,
                "pandas_output_parser": pandas_output_parser,
                "response_synthesis_prompt": response_synthesis_prompt,
                "llm2": self.llm,
            },
        )

        self._configure_pipeline_links()

    def _configure_pipeline_links(self) -> None:
        self.pipeline.add_chain([
            "input",
            "pandas_prompt",
            "llm1",
            "pandas_output_parser"
        ])

        self.pipeline.add_links([
            Link("input", "response_synthesis_prompt", dest_key="query_str"),
            Link("llm1", "response_synthesis_prompt", dest_key="pandas_instructions"),
            Link(
                "pandas_output_parser",
                "response_synthesis_prompt",
                dest_key="pandas_output"
            ),
        ])

        self.pipeline.add_link("response_synthesis_prompt", "llm2")

    def query(self, query: str) -> str:
        try:
            response, intermediate = self.pipeline.run_with_intermediates(query_str=query)
            python_code = intermediate['pandas_output_parser'].inputs['input'].message.blocks[0].text
            return response.message.content, python_code
        except Exception as e:
            raise

