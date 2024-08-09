# Using python image
FROM "python:3.10.14-slim"
RUN apt-get update && apt-get -y install sudo

# Set the current working directory
WORKDIR /app

# Set working from directory
COPY api/api.py /app/api.py
COPY api/requirements.txt /app/requirements.txt
COPY credit_score_mlops /app/credit_score_mlops
COPY params.yaml /app/params.yaml
COPY api.yaml /app/api.yaml
COPY README.md /app/README.md
COPY pyproject.toml /app/pyproject.toml
COPY setup.cfg /app/setup.cfg
COPY Makefile /app/Makefile

# Run make requirements
RUN sudo apt-get -y install make
RUN make requirements

# Expose port 80 to the outside world
EXPOSE 8080

# Run FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
