#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Example TorchGeneratorAgent model.

This demonstrates the minimal structure of a building a generative model, consisting of
an encoder and decoder that each contain a 1-layer LSTM. The subclassed model and agent
handle passing in the current decoder state during incremental decoding, as well as
reordering the encoder/decoder states. The base TorchGeneratorAgent class handles common
generator features like forced decoding, beam search, n-gram beam blocking, and top-k
and top-p/nucleus sampling.

You can train this agent to a reasonable accuracy with:

.. code-block:: bash

    parlai train_model -m examples/seq2seq \
        -mf /tmp/example_model \
        -t convai2 -bs 32 -eps 2 --truncate 128

Afterwards, you can play with --beam-size to see how responses differ with
different beam lengths.
"""  # noqa: E501

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
import torch.nn as nn
import torch.nn.functional as F
import parlai.core.torch_generator_agent as tga


class Encoder(nn.Module):
    """Encodes a sequence of word embeddings"""
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        
    def forward(self, x, mask, lengths):
        """
        Applies a bidirectional LSTM to sequence of embeddings x.
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        packed = pack_padded_sequence(x, lengths, batch_first=True)
        output, (final, _) = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)

        # we need to manually concatenate the final states for both directions
        fwd_final = final[0:final.size(0):2]
        bwd_final = final[1:final.size(0):2]
        final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final

class Decoder(nn.Module):
     """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, attention, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
                 
        self.rnn = nn.LSTM(emb_size + 2*hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.bridge = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
                                          hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_mask, proj_key, hidden):
        """Perform a single decoder step (1 word)"""

        # compute context vector using attention mechanism
        query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, mask=src_mask)

        # update rnn hidden state
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, (final, _) = self.rnn(rnn_input, (hidden, torch.zeros(hidden.size())))
        # output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output, hidden, pre_output
    
    def forward(self, trg_embed, encoder_hidden, encoder_final, 
                src_mask, trg_mask, hidden=None, max_len=None):
        """Unroll the decoder one step at a time."""
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg_mask.size(-1)

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
        
        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        
        # here we store all intermediate hidden states and pre-output vectors
        decoder_states = []
        pre_output_vectors = []
        
        # unroll the decoder RNN for max_len steps
        for i in range(max_len):
            prev_embed = trg_embed[:, i].unsqueeze(1)
            output, hidden, pre_output = self.forward_step(
              prev_embed, encoder_hidden, src_mask, proj_key, hidden)
            decoder_states.append(output)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)
        return decoder_states, hidden, pre_output_vectors  # [B, N, D]

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return torch.tanh(self.bridge(encoder_final))            


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) attention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, mask=None):
        assert mask is not None, "mask is required"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        scores.data.masked_fill_(mask == 0, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas


def make_model(src_vocab, tgt_vocab, emb_size=256, hidden_size=512, num_layers=1, dropout=0.1):
    "Helper: Construct a model from hyperparameters."

    attention = BahdanauAttention(hidden_size)

    model = EncoderDecoder(
        Encoder(emb_size, hidden_size, num_layers=num_layers, dropout=dropout),
        Decoder(emb_size, hidden_size, attention, num_layers=num_layers, dropout=dropout),
        nn.Embedding(src_vocab, emb_size),
        nn.Embedding(tgt_vocab, emb_size),
        Generator(hidden_size, tgt_vocab))

    return model.cuda() if USE_CUDA else model

class Ju2Jo(tga.TorchGeneratorModel):
    """
    Ju2Jo implements the abstract methods of TorchGeneratorModel to define how to
    re-order encoder states and decoder incremental states.
    It also instantiates the embedding table, encoder, and decoder, and defines the
    final output layer.
    """

    def __init__(self, dictionary, hidden_size=1024, num_layers=1, dropout=0):
        super().__init__(
            padding_idx=dictionary[dictionary.null_token],
            start_idx=dictionary[dictionary.start_token],
            end_idx=dictionary[dictionary.end_token],
            unknown_idx=dictionary[dictionary.unk_token],
        )
        attention = BahdanauAttention(hidden_size)
        self.embeddings = nn.Embedding(len(dictionary), hidden_size)
        self.encoder = Encoder(self.embeddings, hidden_size, num_layers=num_layers, dropout=dropout) #Encoder(self.embeddings, hidden_size)
        self.decoder = Decoder(self.embeddings, hidden_size, attention, num_layers=num_layers, dropout=dropout) #Decoder(self.embeddings, hidden_size)

    def output(self, decoder_output):
        """
        Perform the final output -> logits transformation.
        """
        return F.linear(decoder_output, self.embeddings.weight)

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states to select only the given batch indices.
        Since encoder_state can be arbitrary, you must implement this yourself.
        Typically you will just want to index select on the batch dimension.
        """
        h, c = encoder_states
        return h[:, indices, :], c[:, indices, :]

    def reorder_decoder_incremental_state(self, incr_state, indices):
        """
        Reorder the decoder states to select only the given batch indices.
        This method can be a stub which always returns None; this will result in the
        decoder doing a complete forward pass for every single token, making generation
        O(n^2). However, if any state can be cached, then this method should be
        implemented to reduce the generation complexity to O(n).
        """
        h, c = incr_state
        return h[:, indices, :], c[:, indices, :]


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
