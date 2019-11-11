""" [summary]
"""
import argparse
import numpy as np
import pytorch as torch
import torch.nn as nn
from typing import Tuple

from structures.mol_features import N_ATOM_FEATS, N_BOND_FEATS

__author__ = "David Longo (longodj@gmail.com)"


class GraphConvNet(nn.Module):
    """GraphConvNet [summary].

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self,
                 args: argparse.Namespace):
        super(GraphConvNet, self).__init__()
        self.args = args

        """Message Passing Network."""
        self.W_msg_i = nn.Linear(N_ATOM_FEATS +
                                 N_BOND_FEATS,
                                 args.hidden_size,
                                 bias=False)
        
        if args.no_share:
            self.W_msg_h = nn.ModuleList(
                [
                    nn.Linear(args.hidden_size, 
                              args.hidden_size, 
                              bias=False)
                    for _ in range(args.depth - 1)
                ]
            )
        else:
            self.W_msg_h = nn.Linear(args.hidden_size, 
                                     args.hidden_size, 
                                     bias=False)
        
        self.W_msg_o = nn.Linear(N_ATOM_FEATS + args.hidden_size,
                                 args.hidden_size)
        
        """Dropout."""
        self.dropout = nn.Dropout(args.dropout)

    """ [summary]

    Returns
    -------
    [type]
        [description]
    """
    def forward(self, 
                graph_inputs: Tuple[torch.Tensor,   # fatoms
                                    torch.Tensor,   # fbonds
                                    torch.Tensor,   # agraph
                                    torch.Tensor]   # bgraph
                ) -> torch.Tensor:
        
        fatoms, fbonds, agraph, bgraph = graph_inputs

        atom_input = torch.tensor(None)
        atom_input = self.dropout(atom_input)

        atom_h = nn.ReLU()(self.W_message_o(atom_input))
        return atom_h