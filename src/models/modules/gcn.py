""" [summary]
"""
import argparse
import dgl
import dgl.function as fn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from utils.chem import get_mol

import structures.mol_features as mf
from structures.mol_features import N_ATOM_FEATS, N_BOND_FEATS
import structures

__author__ = "David Longo (longodj@gmail.com)"

def mol2dgl_enc(smiles):
    n_edges = 0

    atom_x = []
    bond_x = []

    mol = get_mol(smiles)
    n_atoms = mol.GetNumAtoms()
    n_bonds = mol.GetNumBonds()
    graph = dgl.DGLGraph()
    for i, atom in enumerate(mol.GetAtoms()):
        assert i == atom.GetIdx()
        atom_x.append(mf.get_atom_features(atom))
    graph.add_nodes(n_atoms)

    bond_src = []
    bond_dst = []
    for i, bond in enumerate(mol.GetBonds()):
        begin_idx = bond.GetBeginAtom().GetIdx()
        end_idx = bond.GetEndAtom().GetIdx()
        features = mf.get_bond_features(bond)
        bond_src.append(begin_idx)
        bond_dst.append(end_idx)
        bond_x.append(features)
        # set up the reverse direction
        bond_src.append(end_idx)
        bond_dst.append(begin_idx)
        bond_x.append(features)
    graph.add_edges(bond_src, bond_dst)

    n_edges += n_bonds
    return graph, torch.stack(atom_x), \
            torch.stack(bond_x) if len(bond_x) > 0 else torch.zeros(0)


mpn_loopy_bp_msg = fn.copy_src(src='msg', out='msg')
mpn_loopy_bp_reduce = fn.sum(msg='msg', out='accum_msg')

class LoopyBeliefProp_Update(nn.Module):
    def __init__(self, args):
        super(LoopyBeliefProp_Update, self).__init__()
        self.device = args.device
        self.use_cuda = args.use_cuda
        self.hidden_size = args.hidden_size
        
        self.W_h = nn.Linear(  # y = xA^T + b
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False
        )
        if self.use_cuda: self.W_h = self.W_h.cuda()
    
    def forward(self, nodes):
        msg_input = nodes.data['msg_input']
        accum_msg = nodes.data['accum_msg'].cuda() if self.use_cuda else nodes.data['accum_msg']
        msg_delta = self.W_h(nodes.data['accum_msg'])
        msg = F.relu(msg_input + msg_delta)
        return {'msg': msg}
    
mpn_gather_msg = fn.copy_edge(edge='msg', out='msg')
mpn_gather_reduce = fn.sum(msg='msg', out='m')

class MPN_Gather_Update(nn.Module):
    def __init__(self, args):
        super(MPN_Gather_Update, self).__init__()
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.use_cuda = args.use_cuda
        
        self.W_o = nn.Linear(N_ATOM_FEATS + self.hidden_size, self.hidden_size)
        if self.use_cuda: self.W_o = self.W_o.cuda()
        
    def forward(self, nodes):
        m, x = nodes.data['m'], nodes.data['x']
        if self.use_cuda: 
            m, x = m.cuda(), x.cuda()
        h = F.relu(self.W_o(torch.cat([x, m], 1)))
        if self.use_cuda: h = h.cuda()
            
        return {
            'h': h
        }

class GraphConvNet(nn.Module):
    """GraphConvNet [summary].

    Parameters
    ----------
    nn : [type]
        [description]
    """
    def __init__(self,
                 args):#): argparse.Namespace):
        super(GraphConvNet, self).__init__()
        self.args = args
        self.depth = args.depth
        self.device = args.device
        self.hidden_size = args.hidden_size
        self.n_samples_total = 0
        self.n_nodes_total = 0
        self.n_edges_total = 0
        self.n_passes = 0
        self.use_cuda = args.use_cuda

        """Message Passing Network."""
        self.W_msg_i = nn.Linear(N_ATOM_FEATS +
                                 N_BOND_FEATS,
                                 args.hidden_size,
                                 bias=False)
        
        if self.use_cuda: self.W_msg_i = self.W_msg_i.cuda()
        
        self.apply_mod = LoopyBeliefProp_Update(args)
        self.gather_updater = MPN_Gather_Update(args)

        
        """Dropout."""
        #self.dropout = nn.Dropout(args.dropout)

    """ [summary]

    Returns
    -------
    [type]
        [description]
    """
    def forward(self, 
                mol_graph: structures.MolTree):
        
        mol_line_graph = mol_graph.line_graph(backtracking=False,
                                              shared=True)
        
        n_edges = mol_graph.number_of_edges()
        n_nodes = mol_graph.number_of_nodes()
        n_samples = mol_graph.batch_size
        
        """Run."""
        mol_graph.apply_edges(
            func=lambda edges: {'src_x': edges.src['x']},
        )

        e_repr = mol_line_graph.ndata
        bond_features = e_repr['x']  # torch.Tensor
        source_features = e_repr['src_x']  # torch.tensor
        
        if self.use_cuda: 
            bond_features = bond_features.cuda()
            source_features = source_features.cuda()

        features = torch.cat([source_features, bond_features], 1)
        if self.use_cuda: features = features.cuda()
        msg_input = self.W_msg_i(features)
        mol_line_graph.ndata.update({
            'msg_input': msg_input,
            'msg': F.relu(msg_input),
            'accum_msg': torch.zeros_like(msg_input),
        })
        mol_graph.ndata.update({
            'm': bond_features.new(n_nodes, self.hidden_size).zero_(),
            'h': bond_features.new(n_nodes, self.hidden_size).zero_(),
        })

        for i in range(self.depth - 1):
            mol_line_graph.update_all(
                mpn_loopy_bp_msg,
                mpn_loopy_bp_reduce,
                self.apply_mod,
            )

        mol_graph.update_all(
            mpn_gather_msg,
            mpn_gather_reduce,
            self.gather_updater,
        )

        g_repr = dgl.mean_nodes(mol_graph, 'h')
        
        self.n_samples_total += n_samples
        self.n_nodes_total += n_nodes
        self.n_edges_total += n_edges
        self.n_passes += 1
        
        return g_repr