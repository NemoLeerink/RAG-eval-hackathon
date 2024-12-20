import re
from mlflow.metrics import MetricValue, make_metric
from mlflow.metrics.genai import make_genai_metric_from_prompt

def eval_fn_url(predictions, targets):

    pattern = re.compile("\bhttps?://[^\s]+")
    scores = ["yes" if pattern.match(pred) else "no" for pred in predictions]

    return MetricValue(scores=scores,)

def eval_fn_email(predictions, targets):

    pattern = re.compile("^([A-Z][0-9]+)+$")
    print(predictions)
    scores = ["yes" if pattern.match(pred) else "no" for pred in predictions]

    return MetricValue(scores=scores,)

# Define a custom assessment to detect PII in the retrieved chunks. 
has_pii_prompt = "Your task is to determine whether the retrieved content has any PII information. This was the content: '{retrieved_context}'"


class RetrievalMetrics():
    # Create EvaluationMetric objects during initialization
    email_metric = make_metric(
        eval_fn=eval_fn_email, greater_is_better=False, name="no_exposed_emails"
    )
    url_metric = make_metric(
        eval_fn=eval_fn_url, greater_is_better=False, name="no_exposed_url"
    )

    has_pii = make_genai_metric_from_prompt(
        name="has_pii",
        judge_prompt=has_pii_prompt,
        model="endpoints:/databricks-meta-llama-3-1-70b-instruct",
        metric_metadata={"assessment_type": "RETRIEVAL"},
    )



    
