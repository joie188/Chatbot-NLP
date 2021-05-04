from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import torch.nn as nn
import torch.nn.functional as F
import parlai.core.torch_generator_agent as tga

class Seq2seqAgent(tga.TorchGeneratorAgent):

    @classmethod
    def add_cmdline_args(cls, argparser, partial_opt=None):
        super().add_cmdline_args(argparser, partial_opt=partial_opt)
        group = argparser.add_argument_group('Example TGA Agent')
        group.add_argument(
            '-hid', '--hidden-size', type=int, default=1024, help='Hidden size.')
        group.add_argument('-nl', '--num-layers', type=int, default=1, help='Number of layers in Encoder and Decoder.')
        group.add_argument('-d', '--dropout', type=float, default=0., help='Dropout.')

    def build_model(self):
        model = ExampleModel(self.dict, self.opt['hidden_size'])
        if self.opt['embedding_type'] != 'random':
            self._copy_embeddings(
                model.embeddings.weight, self.opt['embedding_type']
            )
        return model
        