FROM gcr.io/tpu-pytorch/xla:r2.0_3.8_tpuvm

RUN bash -c "source ~/.bashrc && pip install poetry"

# Install dependencies
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN bash -c "source ~/.bashrc && conda activate pytorch && poetry export -f requirements.txt --output requirements.txt"
RUN bash -c "source ~/.bashrc && conda activate pytorch && pip install -r requirements.txt"

# Copy source code
COPY . .
