import gc
import logging
import pickle
import time

import numpy as np
from mapd.probes.make_probe_suites import make_probe_suites
from mapd.utils.make_dataloaders import make_dataloaders
from mapd.utils.wrap_dataset import wrap_dataset

from src.models.mapd_train import get_datasets, get_dataloaders, run_proxies, run_probes, run_without


def benchmark():
    NUM_WORKERS = 16
    BATCH_SIZE = 512
    PREFETCH_FACTOR = 4
    EPOCHS = 5
    N = 10

    logger = logging.getLogger(__name__)

    train_dataset, val_dataset = get_datasets()
    idx_train_dataset = wrap_dataset(train_dataset)
    idx_val_dataset = wrap_dataset(val_dataset)

    logger.info("Starting benchmarking")

    proxy_train_dataloader, validation_dataloader = get_dataloaders(idx_train_dataset, idx_val_dataset, BATCH_SIZE,
                                                                    NUM_WORKERS, PREFETCH_FACTOR)
    proxy_times = []
    for i in range(N):
        logger.info(f"Starting benchmarking iteration {i}")

        start = time.perf_counter()
        run_proxies(proxy_train_dataloader, validation_dataloader, EPOCHS, False, barebones=True)
        end = time.perf_counter()
        gc.collect()

        logger.info(f"Finished benchmarking iteration {i} in {end - start:0.4f} seconds")
        proxy_times.append(end - start)

    logger.info(f"Proxies: {np.mean(proxy_times):.5f} +/- {np.std(proxy_times):.5f}")

    with open("models/proxy-benchmarks.pkl", "wb") as f:
        pickle.dump(proxy_times, f)

    del proxy_train_dataloader, validation_dataloader
    gc.collect()

    logger.info("creating probes")
    train_probes_dataset = make_probe_suites(
        idx_train_dataset, proxy_calculator="models/proxies_e2e", label_count=1000
    )

    probe_train_dataloader, validation_dataloader = get_dataloaders(train_probes_dataset, idx_val_dataset, BATCH_SIZE,
                                                                    NUM_WORKERS, PREFETCH_FACTOR)

    mapd_validation_dataloaders = make_dataloaders(
        [validation_dataloader],
        train_probes_dataset,
        dataloader_kwargs={
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "prefetch_factor": PREFETCH_FACTOR,
        },
    )

    probe_times = []
    for i in range(N):
        logger.info(f"Starting benchmarking iteration {i}")

        start = time.perf_counter()
        run_probes(probe_train_dataloader, mapd_validation_dataloaders, EPOCHS, False, barebones=True)
        end = time.perf_counter()
        gc.collect()

        logger.info(f"Finished benchmarking iteration {i} in {end - start:0.4f} seconds")
        probe_times.append(end - start)

    logger.info(f"Probes: {np.mean(probe_times):.5f} +/- {np.std(probe_times):.5f}")

    with open("models/probe-benchmarks.pkl", "wb") as f:
        pickle.dump(probe_times, f)

    train_dataloader, validation_dataloader = get_dataloaders(train_dataset, val_dataset, BATCH_SIZE,
                                                                    NUM_WORKERS, PREFETCH_FACTOR)

    without_times = []
    for i in range(N):
        logger.info(f"Starting benchmarking iteration {i}")

        start = time.perf_counter()
        run_without(train_dataloader, validation_dataloader, EPOCHS)
        end = time.perf_counter()
        gc.collect()

        logger.info(f"Finished benchmarking iteration {i} in {end - start:0.4f} seconds")
        without_times.append(end - start)

    logger.info(f"Without: {np.mean(without_times):.5f} +/- {np.std(without_times):.5f}")

    with open("models/without-benchmarks.pkl", "wb") as f:
        pickle.dump(without_times, f)

if __name__ == "__main__":
    benchmark()