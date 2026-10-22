FROM python:3.12-slim
WORKDIR /app
RUN python -m pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt pyproject.toml README.md ./
COPY renormcell ./renormcell
COPY data ./data
RUN python -m pip install --no-cache-dir -e .
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "renormcell.predict"]
