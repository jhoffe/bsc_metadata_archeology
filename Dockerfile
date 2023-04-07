FROM gcr.io/tpu-pytorch/xla:r2.0_3.8_tpuvm

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock

RUN poetry install

# Copy source code
COPY . .
