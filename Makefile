.PHONY: setup lint test demo serve docker

setup:
	pip install -e ".[dev,api]"
	pre-commit install || true

lint:
	ruff check . && ruff format --check .

test:
	pytest --cov=claimvalue --cov-report=term-missing

# Full synthetic pipeline: generate -> anonymize -> SQL/ABT -> train -> anomalies
demo:
	claimvalue pipeline --n 20000

serve:
	uvicorn claimvalue.api:app --reload

docker:
	docker build -t claimvalue-api .
