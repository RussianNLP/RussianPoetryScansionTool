import os
from importlib import resources
from importlib.resources import files

from .poetry_alignment import PoetryStressAligner
from .udpipe_parser import UdpipeParser
from .phonetic import Accents



def create_rpst_instance(models_dir: str=None) -> PoetryStressAligner:
    """
    This function loads all the necessary models and dictionaries,
    creates an RPST instance with default settings and returns it.

    By default the models and dictionary are loaded from module installation directory.
    You can path the path to this directory explicitly via `models_dir`.
    """
    if models_dir is None:
        models_dir = files("russian_scansion.models").joinpath('').__str__()

    parser = UdpipeParser()
    parser.load(models_dir)

    accents = Accents(device="cpu")
    accents.load_pretrained(os.path.join(models_dir, 'accentuator'))

    aligner = PoetryStressAligner(parser, accents, model_dir=os.path.join(models_dir, 'scansion_tool'))
    aligner.max_words_per_line = 14

    return aligner
