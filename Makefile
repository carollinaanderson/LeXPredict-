.PHONY: setup lint test demo serve docker explain

setup:
	pip install -e ".[dev,api]"
	pre-commit install || true

lint:
	ruff check . && ruff format --check .

test:
	pytest --cov=claimvalue --cov-report=term-missing

# Full synthetic pipeline: generate -> anonymize -> SQL/ABT -> train -> SHAP -> anomalies
demo:
	claimvalue pipeline --n 20000

explain:
	claimvalue explain --task both --max-cases 800

serve:
	uvicorn claimvalue.api:app --reload

docker:
	docker build -t claimvalue-api .
