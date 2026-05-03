# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# HF Spaces requires the process to run as uid 1000
RUN useradd -m -u 1000 user

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# ── app.py & requirements.txt live in the Space repo ─────────────────────────
COPY app.py requirements.txt .

# ── Clone training repo for shared modules ────────────────────────────────────
RUN git clone --branch feat-cddb --depth 1 \
    https://github.com/aitf-its-tim3-dfk/david /tmp/david && \
    cp /tmp/david/model.py   . && \
    cp /tmp/david/encoder.py . && \
    cp -r /tmp/david/clip    . && \
    rm -rf /tmp/david

# ── Python dependencies ───────────────────────────────────────────────────────
RUN pip install --no-cache-dir -r requirements.txt

# ── Download checkpoints at build time (so cold starts are instant) ───────────
RUN python -m gdown \
    "https://drive.google.com/uc?id=1Nx30Kbu5xnv6dPwz4I3Ivy380LCdp1Md" \
    -O k400_clip_complete_finetuned_30_epochs.pth

RUN python -m gdown \
    "https://drive.google.com/uc?id=1ZIkT6G9UuzqkAj5bpyMHYD9gPonM6EfB" \
    -O best_detector_model.pt

# ── Permissions ───────────────────────────────────────────────────────────────
RUN chown -R 1000:1000 /app

USER user
ENV HOME=/home/user
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "app.py"]
