from __future__ import print_function
import configparser
from models.skip_thoughts.skip_thoughts import configuration, encoder_manager

import os
from urllib.request import urlretrieve
import tarfile

from DownloadProgress import DownloadProgress

class SentenceEncoderBiSkip():
    def __init__(self):
        config = configparser.ConfigParser()
        config.read('.models')
        self.model_params_url = config.get('skip_thoughts_bi', 'model_params_url')
        self.model_params_dir = config.get('skip_thoughts_bi', 'model_params_dir')
        self.model_params_file = config.get('skip_thoughts_bi', 'model_params_file')

        self.VOCAB_FILE = config.get('skip_thoughts_bi', 'VOCAB_FILE')
        self.EMBEDDING_MATRIX_FILE = config.get('skip_thoughts_bi', 'EMBEDDING_MATRIX_FILE')
        self.CHECKPOINT_PATH = config.get('skip_thoughts_bi', 'CHECKPOINT_PATH')

        self.model = None
        self.input_ = None

    def download_params(self):
        if not os.path.exists(self.model_params_dir):
            os.makedirs(self.model_params_dir)

        model_params_fp = os.path.join(self.model_params_dir, self.model_params_file)
        unzipped_path = "".join(model_params_fp.split(".")[:-2])

        if os.path.isdir(unzipped_path):
            print("unzipped directory for skip_thoughts_bi parameters exists; skipping download")
        elif os.path.isfile(model_params_fp):
            print("zip file of skip_thoughts_bi parameters exists; skipping download")
        else:
            with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='skip_thoughts_bi parameters') as progressbar:
                urlretrieve(
                    self.model_params_url,
                    model_params_fp,
                    progressbar.hook)

        # unzip it, if it hasn't been unzipped yet
        # (this also applies if we didn't download the zipfile just now)
        if not os.path.isdir(unzipped_path):
            print("extracting skip_thoughts_bi parameters from zip file...")
            with tarfile.open(model_params_fp) as tar:
                tar.extractall(path=self.model_params_dir)
                tar.close()

    def materialize(self, encoder_mgr=None):
        self.download_params()

        if encoder_mgr is None:
            encoder_mgr = encoder_manager.EncoderManager()

        encoder_mgr.load_model(configuration.model_config(bidirectional_encoder=True),
                           vocabulary_file=os.path.join(self.model_params_dir, self.VOCAB_FILE),
                           embedding_matrix_file=os.path.join(self.model_params_dir, self.EMBEDDING_MATRIX_FILE),
                           checkpoint_path=os.path.join(self.model_params_dir, self.CHECKPOINT_PATH))

        self.model = encoder_mgr

