.PHONY: install lint format test ingest api ui eval mlflow docker-up docker-down

install:
	uv sync --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest --cov

fetch-data:
	uv run python scripts/fetch_edgar.py
	uv run python scripts/make_scanned.py

ingest:
	uv run python -m docsense.ingestion.pipeline --all

api:
	uv run uvicorn docsense.api.main:app --reload --port 8010

ui:
	uv run streamlit run src/docsense/ui/app.py

eval:
	uv run python -m docsense.rag.eval

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down
