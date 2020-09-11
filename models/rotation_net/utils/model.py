import logging
import os

import torch

logger = logging.getLogger('RotationNet')


def save_model(arch, model, optimizer, fname):
    """
    Save the pytorch model

    Parameters
    ----------
    arch : Model architecture (str)
    model : Pytorch model
    optimizer : Pytorch optimizer
    fname : file name to save

    Returns
    -------
    Nothing
    """
    torch.save({
        'arch': arch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, fname)


def load_model(model, path, map_location=None):
    """
    Load model weights for evaluation

    Parameters
    ----------
    model : Pytorch model to load weights
    path : Path to saved model weights

    Returns
    -------
    Nothing, the weights are updated in the given model
    """
    if os.path.isfile(path):
        logger.debug("Loading model '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_location)
        model.load_state_dict(checkpoint['state_dict'])
        logger.debug("Loaded model ({}) '{}'".format(checkpoint['arch'], path))
        del checkpoint
        torch.cuda.empty_cache()
    else:
        logger.warning("No model found at '{}'".format(path))
