# Using python image
FROM "python:3.10.14-slim"
RUN apt-get update && apt-get -y install sudo

# Set the current working directory
WORKDIR /app

# Set working from directory
COPY ./streamlit_deployment/app.py /app/app.py
COPY ./streamlit_deployment/requirements.txt /app/requirements.txt
COPY ./Makefile /app/Makefile

# Run make requirements
RUN sudo apt-get -y install make
RUN make requirements

# Expose port 8080 to the outside world
EXPOSE 8501

# Run FastAPI app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]