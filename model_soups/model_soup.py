import os
from argparse import Namespace
from collections import OrderedDict
import torch

from model import get_model_and_kwargs_for_args

MODEL_1 = 'path to model 1'
MODEL_2 = 'path to model 2'
MODEL_3 = 'path to model 3'
MODEL_4 = 'path to model 4'
MODEL_5 = 'path to model 5'
MODEL_6 = 'path to model 6'
MODEL_7 = 'path to model 7'
MODEL_8 = 'path to model 8'
MODEL_9 = 'path to model 9'
MODEL_10 = 'path to model 10'
MODEL_10_BETA = 'path to model 10 with F-BETA'
MODEL_11 = 'path to model 11'
MODEL_12 = 'path to model 12'
MODEL_12_BETA = 'path to model 12 with F-BETA'

try:
    from torch.hub import _get_torch_home

    torch_cache_home = _get_torch_home()
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'pytorch_transformers')
try:
    from pathlib import Path

    PYTORCH_PRETRAINED_BERT_CACHE = Path(
        os.getenv('PYTORCH_TRANSFORMERS_CACHE', os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', default_cache_path)))
except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_TRANSFORMERS_CACHE',
                                              os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                        default_cache_path))


def load_model(args: Namespace, model_path: str = None, config=None,
               training: bool = True,
               ) -> torch.nn.Module:
    """Instantiates a pretrained model from parsed argument values.

    Args:
        args: parsed arguments from argv.
        model_path: name of model checkpoint or path to a checkpoint directory.
        training: if True, loads a model with training-specific parameters.
    """

    model_class, model_kwargs = get_model_and_kwargs_for_args(
        args, training=training)

    cache_dir = os.path.join(
        PYTORCH_PRETRAINED_BERT_CACHE,
        'distributed_{}'.format(args.local_rank))

    if model_path:
        # Carrega um MODEL pré-treinado
        model = model_class.from_pretrained(
            model_path,
            num_labels=args.num_labels,
            cache_dir=cache_dir,
            output_hidden_states=True,  # Ensure all hidden states are returned
            **model_kwargs)
    else:
        # Instancia um novo MODEL
        model = model_class(config, **model_kwargs)
    return model


def save_model(model, args: Namespace) -> None:
    """Save a trained model and the associated configuration to output dir."""
    model.save_pretrained(args.output_dir)
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))


def run():
    args = Namespace()
    args.num_labels = 11
    args.local_rank = -1
    args.pooler = 'last'
    args.no_crf = True
    args.freeze_bert = False
    args.lstm_hidden_size = 100
    args.lstm_layers = 1
    args.do_train = False
    args.output_dir = 'large_crf_sopinha_do_original_com_o_pre_treinado_com_mlm_mais_data_augmentation_best'

    model1 = load_model(args,
                        model_path=MODEL_10,
                        training=args.do_train)
    model1.to('cuda')

    model2 = load_model(args,
                        model_path=MODEL_11,
                        training=args.do_train)
    model2.to('cuda')

    model3 = load_model(args,
                        MODEL_12,
                        training=args.do_train)
    model3.to('cuda')

    model_soup_3(args, model1, model2, model3)


def run2():
    args = Namespace()
    args.num_labels = 11
    args.local_rank = -1
    args.pooler = 'last'
    args.no_crf = True
    args.freeze_bert = False
    args.lstm_hidden_size = 100
    args.lstm_layers = 1
    args.do_train = False

    #args.output_dir = 'base_crf_sopinha_do_original_com_data_augmentation_best'
    args.output_dir = 'base_MODEL_10_MODEL_12_beta'

    model1 = load_model(args,
                        model_path=MODEL_10_BETA,
                        training=args.do_train)
    model1.to('cuda')

    model2 = load_model(args,
                        MODEL_12_BETA,
                        training=args.do_train)
    model2.to('cuda')

    model_soup_2(args, model1, model2)


def model_soup_2(args, model1, model2):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    assert state_dict1.keys() == state_dict2.keys()
    state_dict = OrderedDict()
    for key in state_dict1.keys():
        state_dict[key] = (state_dict1[key] + state_dict2[key]) / 2
    new_model = load_model(args, config=model1.config, training=args.do_train)
    new_model.load_state_dict(state_dict)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    save_model(new_model, args)


def model_soup_3(args, model1, model2, model3):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    state_dict3 = model3.state_dict()
    assert state_dict1.keys() == state_dict2.keys() == state_dict3.keys()
    state_dict = OrderedDict()
    for key in state_dict1.keys():
        state_dict[key] = (state_dict1[key] + state_dict2[key] + state_dict3[key]) / 3
    new_model = load_model(args, config=model1.config, training=args.do_train)
    new_model.load_state_dict(state_dict)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    save_model(new_model, args)

def model_soup_3_with_median(args, model1, model2, model3):
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    state_dict3 = model3.state_dict()
    assert state_dict1.keys() == state_dict2.keys() == state_dict3.keys()
    state_dict = OrderedDict()
    for key in state_dict1.keys():
        state_dict[key] = torch.median([state_dict1[key], state_dict2[key], state_dict3[key]])

    new_model = load_model(args, config=model1.config, training=args.do_train)
    new_model.load_state_dict(state_dict)
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    save_model(new_model, args)


if __name__ == '__main__':
    run2()
