# The official MLflow server with the one package its login (basic-auth) app needs,
# pinned. The LLM judges call OpenAI through MLflow's own client, so nothing else
# is added.
FROM ghcr.io/mlflow/mlflow:v3.17.2
RUN pip install --no-cache-dir "Flask-WTF==1.3.0"
