# The inference UI, as one image: node builds the React bundle, python serves
# it next to the model.
#
# Nothing large is baked in. The weights (~8 GB), the X-rays and the LoRA
# adapter are bind-mounted at run time, so the image stays a few GB and a new
# adapter does not mean a rebuild.
#
#     docker compose -f docker/compose.yaml build
#     docker compose -f docker/compose.yaml run --rm fetch-model
#     docker compose -f docker/compose.yaml up serve
#
# TORCH_INDEX picks the wheel: CPU by default, because that is what a laptop
# has. On a GPU box build with
#     --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124

# --------------------------------------------------------------------------
# 1. the React bundle
# --------------------------------------------------------------------------
FROM node:20-slim AS ui-build
WORKDIR /ui

# package.json alone first: the dependency layer is then cached across every
# edit to the application source, which is the part that actually changes.
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY ui/ ./
RUN npm run build


# --------------------------------------------------------------------------
# 2. the runtime
# --------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/.hf

WORKDIR /app

# torch comes from its own index, pinned to the build's flavour; the rest of
# requirements.txt then resolves against PyPI and finds torch already present.
RUN pip install --index-url "${TORCH_INDEX}" "torch>=2.4"

COPY requirements.txt ./
RUN pip install -r requirements.txt huggingface_hub

# The package last, so a code change rebuilds one small layer rather than
# reinstalling eight gigabytes of wheels.
COPY vindr_cxr_vqa/ ./vindr_cxr_vqa/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY --from=ui-build /ui/dist ./ui/dist

EXPOSE 8000

# 0.0.0.0, not 127.0.0.1: inside a container the loopback interface is the
# container's own, and the published port would reach nothing.
CMD ["python", "-m", "vindr_cxr_vqa.serve", "--host", "0.0.0.0", "--port", "8000"]
