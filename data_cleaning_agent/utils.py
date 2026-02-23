# Utility functions for lightweight data cleaning agent

import re
import logging
import numpy as np
import pandas as pd
from langchain_core.output_parsers import BaseOutputParser

logger = logging.getLogger(__name__)


class PythonOutputParser(BaseOutputParser):
    """Extract Python code from LLM responses."""
    
    def parse(self, text: str):
        """Extract code from ```python``` blocks or return text as-is."""
        python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
        if python_code_match:
            return python_code_match.group(1).strip()
        return text


def get_dataframe_summary(df: pd.DataFrame) -> str:
    """
    Generate a simple summary of a DataFrame for the LLM.
    
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize.
    
    Returns
    -------
    str
        A text summary of the DataFrame.
    """
    missing_stats = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    missing_summary = "\n".join([f"{col}: {val:.2f}%" for col, val in missing_stats.items()])
    
    column_types = "\n".join([f"{col}: {dtype}" for col, dtype in df.dtypes.items()])

    data_summary = df.describe().to_string()

    # --- Extra Numeric stats per column
    numeric_stats = df.select_dtypes(include=['number']).columns

    extra_stats = []

    for col in numeric_stats:
        data = df[col].dropna()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        if iqr != 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int((data<lower).sum() + (data>upper).sum())
        else:
            outlier_count = 0

        mode_values = data.mode()

        mode = mode_values.iloc[0] if not mode_values.empty else float("nan")

        extra_stats.append(
            f"{col} | mean = {data.mean():.2f}, median = {data.median():.2f}, mode = {mode:.2f},"
            f"std = {data.std():.2f}, min = {data.min():.2f}, max = {data.max():.2f}, range = {data.max() - data.min():.2f},"
            f"75th percentile = {q3:.2f}, 25th percentile = {q1:.2f}, IQR = {iqr:.2f}, outlier_count = {outlier_count}")

    extra_stats_string = "\n".join(extra_stats) if extra_stats else "No extra numeric stats"
            



    summary = f"""
        Dataset Summary:
        ----------------

        Column Data Types:
        {column_types}

        Missing Value Percentage:
        {missing_summary}
        
        Data Summary:
        {data_summary}
        
        Extra Numeric Stats:
        {extra_stats_string}"""

    return summary.strip()


def execute_agent_code(state, data_key, code_snippet_key, result_key, error_key, agent_function_name):
    """
    Execute the generated agent code on the data.
    
    Parameters
    ----------
    state : dict
        The current state containing data and code.
    data_key : str
        Key in state where the input data is stored.
    code_snippet_key : str
        Key in state where the generated code is stored.
    result_key : str
        Key to store the result in.
    error_key : str
        Key to store any error message in.
    agent_function_name : str
        Name of the function to execute from the generated code.
    
    Returns
    -------
    dict
        Dictionary with result and error keys.
    """
    logger.info("Executing agent code")
    
    data = state.get(data_key)
    agent_code = state.get(code_snippet_key)
    df = pd.DataFrame.from_dict(data)
    
    # Execute the LLM-generated code in isolated namespace
    # Note: exec() can be risky - only use with trusted LLM-generated code
    local_vars = {}
    global_vars = {}
    exec(agent_code, global_vars, local_vars)
    
    # Get the function from executed code
    agent_function = local_vars.get(agent_function_name)
    if not agent_function or not callable(agent_function):
        raise ValueError(f"Function '{agent_function_name}' not found in generated code.")
    
    # Run the function and handle errors
    agent_error = None
    result = None
    try:
        result = agent_function(df)
        if isinstance(result, pd.DataFrame):
            result = result.to_dict()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        agent_error = f"An error occurred during data cleaning: {str(e)}"
    
    return {result_key: result, error_key: agent_error}


def fix_agent_code(state, code_snippet_key, error_key, llm, prompt_template, function_name, retry_count_key="retry_count"):
    """
    Fix errors in the generated agent code using the LLM.
    
    Parameters
    ----------
    state : dict
        The current state containing code and error information.
    code_snippet_key : str
        Key in state where the broken code is stored.
    error_key : str
        Key in state where the error message is stored.
    llm : LLM
        The language model to use for fixing the code.
    prompt_template : str
        Template for the fix prompt (should have {code_snippet}, {error}, {function_name} placeholders).
    function_name : str
        Name of the function being fixed.
    retry_count_key : str, optional
        Key in state for tracking retry count. Defaults to "retry_count".
    
    Returns
    -------
    dict
        Dictionary with updated code, cleared error, and incremented retry count.
    """
    logger.info("Fixing agent code")
    logger.debug(f"Retry count: {state.get(retry_count_key)}")
    
    code_snippet = state.get(code_snippet_key)
    error_message = state.get(error_key)
    
    # Create the fix prompt
    prompt = prompt_template.format(
        code_snippet=code_snippet,
        error=error_message,
        function_name=function_name,
    )
    
    # Get fixed code from LLM
    response = (llm | PythonOutputParser()).invoke(prompt)
    
    return {
        code_snippet_key: response,
        error_key: None,
        retry_count_key: state.get(retry_count_key) + 1
    }

def quality_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate quality metrics for a DataFrame.
    
    Parameters
    ----------
        df : pd.DataFrame
            The DataFrame to calculate quality metrics for.
    
    Returns
    -------
    dict
        A dictionary containing the quality metrics.
    """

    missing_pct = (df.isna().mean() * 100).sort_values(ascending=False)
    dup_rows = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=['number']).columns

    outlier_counts = {}
    for col in numeric_cols:
        data = df[col].dropna()
        if data.empty:
            outlier_counts[col] = 0
            continue
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        if iqr != 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_counts[col] = int((data<lower).sum() + (data>upper).sum())
        else:
            outlier_counts[col] = 0

    row_missing_over_50 = int((df.isna().mean(axis=1) > 0.5).sum())

    return {
        "shape": df.shape,
        "missing_pct": missing_pct,
        "dup_rows": dup_rows,
        "dtype_counts": df.dtypes.astype(str).value_counts(),
        "outlier_counts": pd.Series(outlier_counts).sort_values(ascending=False) if outlier_counts else pd.Series(dtype=int),
        "row_missing_over_50": row_missing_over_50,
    }
