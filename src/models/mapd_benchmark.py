import gc
import logging
import pickle
import time

import numpy as np
import torch
from mapd.probes.make_probe_suites import make_probe_suites
from mapd.utils.make_dataloaders import make_dataloaders
from mapd.utils.wrap_dataset import wrap_dataset

from src.models.mapd_train import get_datasets, get_dataloaders, run_proxies, run_probes, run_without


def benchmark_proxy(num_workers: int, batch_size: int, prefetch_factor: int, epochs: int, n: int):
    logger = logging.getLogger(__name__)

    proxy_times = []
    for i in range(n):
        logger.info(f"Starting benchmarking iteration {i}")

        train_dataset, val_dataset = get_datasets()
        idx_train_dataset = wrap_dataset(train_dataset)
        idx_val_dataset = wrap_dataset(val_dataset)

        logger.info("Starting benchmarking")

        proxy_train_dataloader, validation_dataloader = get_dataloaders(idx_train_dataset, idx_val_dataset, batch_size,
                                                                        num_workers, prefetch_factor)

        start = time.perf_counter()
        run_proxies(proxy_train_dataloader, validation_dataloader, epochs, False, barebones=True,
                    proxy_output_path="models/benchmark_proxies_e2e")
        end = time.perf_counter()

        logger.info(f"Finished benchmarking iteration {i} in {end - start:0.4f} seconds")
        proxy_times.append(end - start)

    logger.info(f"Proxies: {np.mean(proxy_times):.5f} +/- {np.std(proxy_times):.5f}")

    with open("models/proxy-benchmarks.pkl", "wb") as f:
        pickle.dump(proxy_times, f)


def benchmark_probes(num_workers: int, batch_size: int, prefetch_factor: int, epochs: int, n: int):
    logger = logging.getLogger(__name__)

    probe_times = []
    for i in range(n):
        logger.info(f"Starting benchmarking iteration {i}")

        train_dataset, val_dataset = get_datasets()
        idx_train_dataset = wrap_dataset(train_dataset)
        idx_val_dataset = wrap_dataset(val_dataset)

        logger.info("creating probes")
        train_probes_dataset = make_probe_suites(
            idx_train_dataset, proxy_calculator="models/proxies_e2e", label_count=1000
        )

        probe_train_dataloader, validation_dataloader = get_dataloaders(train_probes_dataset, idx_val_dataset,
                                                                        batch_size,
                                                                        num_workers, prefetch_factor)
        mapd_validation_dataloaders = make_dataloaders(
            [validation_dataloader],
            train_probes_dataset,
            dataloader_kwargs={
                "batch_size": batch_size,
                "num_workers": num_workers,
                "prefetch_factor": prefetch_factor,
            },
        )

        start = time.perf_counter()
        run_probes(probe_train_dataloader, mapd_validation_dataloaders, epochs, False, barebones=True,
                   probe_output_path="models/benchmark_probes_e2e")
        end = time.perf_counter()

        logger.info(f"Finished benchmarking iteration {i} in {end - start:0.4f} seconds")
        probe_times.append(end - start)

    logger.info(f"Probes: {np.mean(probe_times):.5f} +/- {np.std(probe_times):.5f}")

    with open("models/probe-benchmarks.pkl", "wb") as f:
        pickle.dump(probe_times, f)


def benchmark_without(num_workers: int, batch_size: int, prefetch_factor: int, epochs: int, n: int):
    logger = logging.getLogger(__name__)
    without_times = []
    for i in range(n):
        logger.info(f"Starting benchmarking iteration {i}")

        train_dataset, val_dataset = get_datasets()

        train_dataloader, validation_dataloader = get_dataloaders(train_dataset, val_dataset, batch_size,
                                                                  num_workers, prefetch_factor)

        start = time.perf_counter()
        run_without(train_dataloader, validation_dataloader, epochs)
        end = time.perf_counter()
        gc.collect()

        logger.info(f"Finished benchmarking iteration {i} in {end - start:0.4f} seconds")
        without_times.append(end - start)

    logger.info(f"Without: {np.mean(without_times):.5f} +/- {np.std(without_times):.5f}")

    with open("models/without-benchmarks.pkl", "wb") as f:
        pickle.dump(without_times, f)


def benchmark():
    NUM_WORKERS = 16
    BATCH_SIZE = 512
    PREFETCH_FACTOR = 4
    EPOCHS = 5
    N = 10

    torch.set_float32_matmul_precision('high')

    benchmark_proxy(NUM_WORKERS, BATCH_SIZE, PREFETCH_FACTOR, EPOCHS, N)
    benchmark_probes(NUM_WORKERS, BATCH_SIZE, PREFETCH_FACTOR, EPOCHS, N)
    benchmark_without(NUM_WORKERS, BATCH_SIZE, PREFETCH_FACTOR, EPOCHS, N)


if __name__ == "__main__":
    benchmark()
