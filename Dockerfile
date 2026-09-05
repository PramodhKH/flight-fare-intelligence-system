FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY api ./api
COPY configs ./configs
EXPOSE 8000
CMD ["python", "-c", "print('API image scaffold ready; FastAPI entrypoint arrives in the deployment phase.')"]
