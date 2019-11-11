"""Molecular Tree.
Adapted from:
******************************************************************
Title: DGL JTNN
Author: Mufei Li (mufeili1996@gmail.com)
Date: September 10, 2019
Code version: 9df8cd3
Availability: https://github.com/dmlc/dgl.git
******************************************************************
"""

import numpy as np
import torch as torch

from dgl import DGLGraph
from rdkit import Chem

from utils.chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles, \
                            set_atommap, enum_assemble_nx, decode_stereo

from .mol_features import get_atom_features, get_bond_features
from .vocab import Vocab

class MolTree(DGLGraph):
    def __init__(self, smiles):
        DGLGraph.__init__(self)
        if smiles is None:
            return

        self.nodes_dict = {}
        self.smiles = smiles
        self.mol = get_mol(smiles)

        """Stereo Generation."""
        mol = Chem.MolFromSmiles(smiles)
        
        self.smiles2D = Chem.MolToSmiles(mol)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.stereo_cands = decode_stereo(self.smiles2D)

        """Cliques: [[Neighbor Indices] for Atom in Atoms]"""
        cliques, edges = tree_decomp(self.mol)
        root = 0
        
        for i, clique in enumerate(cliques):
            cmol = get_clique_mol(self.mol, clique)
            csmiles = get_smiles(cmol)
            self.nodes_dict[i] = dict(
                smiles=csmiles,
                mol=get_mol(csmiles),
                clique=clique)
            if min(clique) == 0:
                root = i

        self.add_nodes(len(cliques))

        if root > 0:
            for attr in self.nodes_dict[0]:
                self.nodes_dict[0][attr] = self.nodes_dict[root][attr]
                self.nodes_dict[root][attr] = self.nodes_dict[0][attr]

        """Generate Edges."""
        src = np.zeros((len(edges) * 2,), dtype='int')
        dst = np.zeros((len(edges) * 2,), dtype='int')
        for  i, (_x, _y) in enumerate(edges):
            x = 0 if _x == root else root if _x == 0 else _x
            y = 0 if _y == root else root if _y == 0 else _y
            src[2*i], dst[2*i] = x, y
            src[2*i+1], dst[2*i+1] = y, x
        
        self.add_edges(src, dst)

        for i in self.nodes_dict:
            self.nodes_dict[i]['nid'] = i+1
            if self.out_degree(i) > 1:
                set_atommap(self.nodes_dict[i]['mol'], self.nodes_dict[i]['nid'])
            self.nodes_dict[i]['is_leaf'] = (self.out_degree(i) == 1)

    @property
    def treesize(self):
        return self.number_of_nodes()
    
    def _recover_node(self, i, original_mol):
        node = self.nodes_dict[i]

        clique = []
        clique.extend(node['clique'])
        if not node['is_leaf']:
            for cidx in node['clique']:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(node['nid'])

        for j in self.successors(i).numpy():
            nei_node = self.nodes_dict[j]
            clique.extend(nei_node['clique'])
            if nei_node['is_leaf']: # Leaf node, no need to mark
                continue
            for cidx in nei_node['clique']:
                # allow singleton node override the atom mapping
                if cidx not in node['clique'] or len(nei_node['clique']) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node['nid'])

        clique = list(set(clique))
        label_mol = get_clique_mol(original_mol, clique)
        node['label'] = Chem.MolToSmiles(Chem.MolFromSmiles(get_smiles(label_mol)))
        node['label_mol'] = get_mol(node['label'])

        for cidx in clique:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return node['label']

    def _assemble_node(self, i):
        neighbors = [self.nodes_dict[j] for j in self.successors(i).numpy()
                     if self.nodes_dict[j]['mol'].GetNumAtoms() > 1]
        neighbors = sorted(neighbors, 
                           key=lambda x: x['mol'].GetNumAtoms(), 
                           reverse=True)
        singletons = [self.nodes_dict[j] for j in self.successors(i).numpy()
                      if self.nodes_dict[j]['mol'].GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cands = enum_assemble_nx(self.nodes_dict[i], neighbors)
        
        if len(cands) > 0:
            self.nodes_dict[i]['cands'], self.nodes_dict[i]['cand_mols'], _ = list(zip(*cands))
            self.nodes_dict[i]['cands'] = list(self.nodes_dict[i]['cands'])
            self.nodes_dict[i]['cand_mols'] = list(self.nodes_dict[i]['cand_mols'])
        else:
            self.nodes_dict[i]['cands'] = []
            self.nodes_dict[i]['cand_mols'] = []

    def recover(self):
        for node in self.nodes_dict:
            self._recover_node(node, self.mol)

    def assemble(self):
        for node in self.nodes_dict:
            self._assemble_node(node)
            
    def encode(self, recssemble=False, vocab=None):
        if recssemble:
            self.recover()
            self.assemble()
        
        wid = None
        if vocab is not None:
            wid = self._set_node_id(self, vocab)
        
        n_edges = 0

        atom_x, bond_x = [], []

        n_atoms, n_bonds = self.mol.GetNumAtoms(), self.mol.GetNumBonds()
        graph = DGLGraph()
        for i, atom in enumerate(self.mol.GetAtoms()):
            assert i == atom.GetIdx()
            atom_x.append(get_atom_features(atom))
        graph.add_nodes(n_atoms)

        bond_src, bond_dst = [], []
        for i, bond in enumerate(self.mol.GetBonds()):
            begin_idx, end_idx = bond.GetBeginAtom().GetIdx(), \
                                 bond.GetEndAtom().GetIdx()
            features = get_bond_features(bond)
            bond_src.append(begin_idx)
            bond_dst.append(end_idx)
            bond_x.append(features)
            # set up the reverse direction
            bond_src.append(end_idx)
            bond_dst.append(begin_idx)
            bond_x.append(features)
        graph.add_edges(bond_src, bond_dst)

        n_edges += n_bonds
        result = graph, \
            torch.stack(atom_x), \
            torch.stack(bond_x) if len(bond_x) > 0 else torch.zeros(0), \
            wid
        return result if vocab else result[0:3] 
                
    @staticmethod
    def _set_node_id(mol_tree, vocab: Vocab):
        wid = []
        for i, node in enumerate(mol_tree.nodes_dict):
            mol_tree.nodes_dict[node]['idx'] = i
            wid.append(vocab.get_index(mol_tree.nodes_dict[node]['smiles']))

        return wid