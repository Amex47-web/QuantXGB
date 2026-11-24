.PHONY: install check lint type test test-fast data features train sanity backtest report score app api mlflow demo docker clean-demo dictionary

RUN := uv run
RUN_ID ?= latest

install:
	uv sync
	$(RUN) pre-commit install

lint:
	$(RUN) ruff check src tests app
	$(RUN) ruff format --check src tests app

type:
	$(RUN) mypy

test:
	$(RUN) pytest -q

test-fast:
	$(RUN) pytest -q -m "not slow"

check: lint type test

data:
	$(RUN) quantxgb data ingest
	$(RUN) quantxgb data build

features:
	$(RUN) quantxgb features build

dictionary:
	$(RUN) quantxgb features dictionary

train:
	@test -n "$(H)" || (echo 'usage: make train H="pre-registered hypothesis"' && exit 1)
	$(RUN) quantxgb train --hypothesis "$(H)"

sanity:
	$(RUN) quantxgb train --hypothesis "Sanity: shuffled labels must give IC ~ 0" --shuffle-labels

backtest:
	$(RUN) quantxgb backtest --run $(RUN_ID)

report:
	$(RUN) quantxgb report --run $(RUN_ID)

score:
	$(RUN) quantxgb score --run $(RUN_ID)

app:
	$(RUN) streamlit run app/Home.py

api:
	$(RUN) uvicorn quantxgb.api.main:app --host 127.0.0.1 --port 8000

mlflow:
	$(RUN) mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --host 127.0.0.1 --port 5000

demo:
	$(RUN) quantxgb demo

clean-demo:
	rm -rf data/demo artifacts/demo

docker:
	docker compose up --build
