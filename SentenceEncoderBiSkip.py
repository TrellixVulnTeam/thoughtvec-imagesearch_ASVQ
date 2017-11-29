import configparser
from models.skip_thoughts.skip_thoughts import configuration, encoder_manager

import os
from urllib.request import urlretrieve
import tarfile

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
        if not os.path.exists(self.model_param_dir):
            os.makedirs(self.model_param_dir)

        model_params_fp = os.path.join(self.model_param_dir, self.model_param_files)
        unzipped_path = "".join(model_params_fp.split(".")[:-2])

        # if we don't have the unzipped dir with the params, and we don't have the zipfile, we will have to get it
        if not os.path.isdir(unzipped_path) and not os.path.isfile(model_params_fp):
            urlretrieve(self.model_params_url, model_params_fp)

        # unzip it, if it hasn't been unzipped yet
        # (this also applies if we didn't download the zipfile just now)
        if not os.path.isdir(unzipped_path):
            with tarfile.open(model_params_fp) as tar:
                tar.extractall()
                tar.close()

    def materialize(self, encoder_mgr=None):
        self.download_params()

        if encoder_mgr is None:
            encoder_mgr = encoder_manager.EncoderManager()

        encoder_mgr.load_model(configuration.model_config(bidirectional_encoder=True),
                           vocabulary_file=self.VOCAB_FILE,
                           embedding_matrix_file=self.EMBEDDING_MATRIX_FILE,
                           checkpoint_path=self.CHECKPOINT_PATH)

        self.model = encoder_mgr

