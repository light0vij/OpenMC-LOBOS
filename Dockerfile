# =============================================================================
# Dockerfile
# OpenMC + JupyterLab — LLM-Orchestrated Bayesian Optimization with Seeding (openmc-lobos)
#
# KEY DECISIONS from build history:
#   - ubuntu:22.04 fixed (not latest — unstable)
#   - OpenMC installed via shimwell pre-built wheels (not compiled from source)
#     saves ~30 min build time
#   - Nuclear data NOT baked into image — downloaded at container startup
#     via executedownload.sh (keeps image lean ~3 GB instead of ~11 GB)
#   - GPy removed — broken on Python 3.11
#   - h5py pinned to 3.10.0 — matches Ubuntu 22.04 libhdf5 (1.10.x)
#   - numpy pinned to 1.26.4
#   - pv installed for extraction progress bar
#   - Notebooks/, environments/, openmc_bo/, sampler_prompts/, Results/ and
#     prompt2.json are all copied in as siblings under /workspace so
#     relative paths inside the notebooks (e.g. "../Results/<model>",
#     "../environments", "../sampler_prompts/<name>.json", "../prompt2.json")
#     resolve the same way they do outside Docker
# =============================================================================

FROM ubuntu:22.04

# ── Suppress interactive apt prompts ─────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV OPENMC_CROSS_SECTIONS=/nuclear_data/cross_sections.xml

# ── Build-time args ───────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.11

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        wget \
        curl \
        pv \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        libhdf5-dev \
        libpng-dev \
        libopenmpi-dev \
        openmpi-bin \
        gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 \
        /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python \
        /usr/bin/python${PYTHON_VERSION} 1

# ── Upgrade pip before installing requirements ───────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# ── Python packages ───────────────────────────────────────────────────────────
# OpenMC is installed via shimwell's pre-built wheels index —
# no need to compile from source, saves ~30 min on every build.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── Workspace ─────────────────────────────────────────────────────────────────
RUN mkdir -p /workspace/Notebooks /workspace/environments /workspace/openmc_bo \
             /workspace/Results /workspace/sampler_prompts
WORKDIR /workspace

# Notebooks — what JupyterLab opens into
COPY Notebooks/    /workspace/Notebooks/

# environments/ — model-specific OpenMC wrappers, imported directly by the
# notebooks via sys.path (see PROJECT_ROOT = os.path.abspath("..")).
COPY environments/  /workspace/environments/

# openmc_bo/ — the optimiser package (LLM sampler, GP/EI, BO loop, LLM
# bounds orchestrator, results). Imported the same way as environments/,
# i.e. `from openmc_bo import ...` 
COPY openmc_bo/      /workspace/openmc_bo/

# sampler_prompts/ — the LLM Generative Seeding prompt configs
# (disperse / E_repulsion / hybrid_E_disperse), read via
# "../sampler_prompts/<name>.json" from inside Notebooks/.
COPY sampler_prompts/ /workspace/sampler_prompts/

# prompt2.json — the LLM Bounds Orchestration prompt config, read via
# "../prompt2.json" from inside Notebooks/.
COPY prompt2.json     /workspace/prompt2.json

# Results/ — pre-existing sample results, plus the directory the notebooks
# write new runs into.
COPY Results/       /workspace/Results/

# ── JupyterLab configuration ──────────────────────────────────────────────────
RUN jupyter lab --generate-config \
    && printf '%s\n' \
        "c.ServerApp.ip = '0.0.0.0'" \
        "c.ServerApp.port = 8888" \
        "c.ServerApp.open_browser = False" \
        "c.ServerApp.token = ''" \
        "c.ServerApp.password = ''" \
        "c.ServerApp.allow_root = True" \
        "c.ServerApp.notebook_dir = '/workspace'" \
       >> /root/.jupyter/jupyter_lab_config.py

# ── Entrypoint ────────────────────────────────────────────────────────────────
# executedownload.sh checks for nuclear data on startup,
# downloads if missing, then starts JupyterLab.
COPY executedownload.sh /executedownload.sh
RUN chmod +x /executedownload.sh

EXPOSE 8888

ENTRYPOINT ["/executedownload.sh"]
