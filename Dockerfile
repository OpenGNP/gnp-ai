# FastAPI served by uvicorn — no build step, so a single stage is enough.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# CPU-only torch wheels: the default Linux wheels pull ~several GB of CUDA libs
# the VPS has no GPU for. The extra index serves torch==X.Y.Z+cpu, which pip
# prefers over the PyPI build; everything else still resolves from PyPI.
COPY requirements.txt ./
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
