# Makefile for the data cleaning agent

.PHONY: run docker stop

run:
	@echo "Starting the data cleaning agent streamlit app..."
	@streamlit run home.py

docker:
	@echo "Building the data cleaning agent docker image..."
	@docker build -t data-cleaning-agent .
	@echo "Stopping and removing existing data cleaning agent docker container..."
	-@docker stop data-cleaning-agent-streamlit-app && docker rm data-cleaning-agent-streamlit-app
	@echo "Running the data cleaning agent docker container..."
	@docker run --name data-cleaning-agent-streamlit-app -p 8501:8501 data-cleaning-agent

stop:
	@echo "Stopping the data cleaning agent docker container..."
	-@docker stop data-cleaning-agent-streamlit-app && docker rm data-cleaning-agent-streamlit-app