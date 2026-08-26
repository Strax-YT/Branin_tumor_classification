FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# api/service.py imports from app/ (direct_model_loader, safety_engine,
# image_quality, model_registry). Deliberately NOT copying models/model2.py:
# it pulls in matplotlib/seaborn/scikit-learn at import time (leftover from
# its training-script role) that aren't worth shipping in this image just to
# define LiteCNN -- direct_model_loader.py already has an identical fallback
# LiteCNN class for exactly this case (verified working this session).
# artifacts/ carries the checkpoint + metadata these all load at startup.
COPY api/ /app/api/
COPY app/direct_model_loader.py app/safety_engine.py app/image_quality.py app/model_registry.py /app/app/
COPY artifacts/ /app/artifacts/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
