FROM python:3.12-slim AS build
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[api]"

FROM python:3.12-slim
RUN useradd --create-home app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
USER app
WORKDIR /home/app
# Models are mounted, never baked into the image. API_KEY is injected at runtime.
ENV MODEL_DIR=/models
EXPOSE 8000
HEALTHCHECK --interval=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
CMD ["uvicorn", "claimvalue.api:app", "--host", "0.0.0.0", "--port", "8000"]
