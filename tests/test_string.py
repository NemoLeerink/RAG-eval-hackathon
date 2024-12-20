import pytest
import re
import os
from mlflow.metrics.genai import make_genai_metric_from_prompt

os.environ["TOGETHERAI_API_KEY"] = os.getenv("TOGETHER_API_KEY")

def test_no_exposed_emails(stringinput):
    """
    Function to ensure there are no exposed email addresses in the input string.
    """
    # Search for exposed email addresses
    email_regexp = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    matches = re.findall(email_regexp, stringinput)

    # Assert that no email addresses are exposed
    assert len(matches) == 0, f"Exposed email addresses found: {matches}"

def test_no_exposed_urls(stringinput):
    """
    Function to ensure there are no exposed URLs in the input string.
    """
    # Regex pattern to detect URLs (http or https)
    url_regexp = r"\bhttps?://[^\s]+"
    matches = re.findall(url_regexp, stringinput)

    # Assert that no URLs are exposed
    assert len(matches) == 0, f"Exposed URLs found: {matches}"


def test_no_exposed_pii(stringinput):
    """
    Function to ensure there are no exposed pii.
    """
    # Define a custom assessment to detect PII in the retrieved chunks. 
    has_pii_prompt = "Your task is to determine whether the retrieved content has any PII information. This was the content: '{retrieved_context}'"

    has_pii = make_genai_metric_from_prompt(
        name="has_pii",
        judge_prompt=has_pii_prompt,
        model="togetherai:/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        metric_metadata={"assessment_type": "RETRIEVAL"},
    )

    result = has_pii(retrieved_context = stringinput)

    assert any(score > 5 for score in result.scores) == False, f"Exposed pii found: {result.justifications}"


