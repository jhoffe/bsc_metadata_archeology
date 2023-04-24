import logging
from pathlib import Path

import click
import pytorch_lightning as pl
import torch.nn
import torchaudio
import torchaudio.functional as F
# import torchaudio.transforms as T
from dotenv import find_dotenv, load_dotenv
from torchaudio.utils import download_asset

from src.data.audio_c_scores import c_scores_dataset as audio_c_scores_dataset
from src.data.make_probe_suites import make_probe_suites


class AudioCorrupter(torch.nn.Module):
    def __init__(self):
        super().__init__()

        SAMPLE_NOISE = download_asset(
            "tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav"
        )
        self.noise, _ = torchaudio.load(SAMPLE_NOISE)

    def forward(self, x):
        noise = self.noise[:, : x.shape[1]]
        snr_dbs = torch.tensor([20, 10, 3])

        return F.add_noise(x, noise, snr_dbs)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("proxy_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, proxy_filepath, output_filepath):
    pl.seed_everything(123)
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final speech commands proxy dataset")

    logger.info("creating dataset with attached scores")
    audio_c_scores_dataset(
        "speechcommands", input_filepath, output_filepath, proxy_filepath
    )

    logger.info("creating probe suite for speechcommands")

    corruption_module = AudioCorrupter()

    make_probe_suites(
        input_filepath,
        output_filepath,
        "speechcommands",
        label_count=35,
        corruption_module=corruption_module,
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
