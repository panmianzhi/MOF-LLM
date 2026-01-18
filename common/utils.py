import torch
from typing import List

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from openbabel import pybel, openbabel as ob

# Disable warnings
RDLogger.DisableLog('rdApp.*')
pybel.ob.obErrorLog.SetOutputLevel(0)


METALS = torch.tensor([
    3, 4, 11, 12, 13, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93, 94,
    95, 96, 97, 98, 99, 100, 101, 102, 103
]).long()


def is_metal_bb(bb_mol: Chem.rdchem.Mol | torch.Tensor) -> bool:
    """
    Returns:
    - bool: True if the molecule contains a metal atom, False otherwise
    """
    if isinstance(bb_mol, Chem.rdchem.Mol):
        bb_atom_types = torch.tensor([atom.GetAtomicNum() for atom in bb_mol.GetAtoms()]).long()
    elif isinstance(bb_mol, torch.Tensor):
        bb_atom_types = bb_mol
    else:
        raise ValueError(f"Invalid type: {type(bb_mol)}. Expected torch.Tensor or RDKit Mol object.")
    return torch.any(torch.isin(bb_atom_types, METALS)).item()


# Helper functions for 3D structure to SMILES conversion (imported from mof_llm.py logic)
def ob_mol_from_data(atom_types: torch.Tensor, positions: torch.Tensor, is_metal: bool):
    """Create OpenBabel molecule from atom types and coordinates"""
    
    atom_types = atom_types.cpu().numpy() if atom_types.is_cuda else atom_types.numpy()
    positions = positions.cpu().numpy() if positions.is_cuda else positions.numpy()
    
    mol = pybel.ob.OBMol()
    for atom_type, position in zip(atom_types, positions):
        atom = pybel.ob.OBAtom()
        atom.SetAtomicNum(int(atom_type))
        atom.SetVector(*position.tolist())
        mol.AddAtom(atom)

    # Organic molecules: automatically infer chemical bonds
    if not is_metal:
        mol.ConnectTheDots()
        for atom in ob.OBMolAtomIter(mol):
            if "N" in atom.GetType() and atom.IsInRing():
                atom.SetAromatic()
        mol.PerceiveBondOrders()
    
    return pybel.Molecule(mol)


def ob_to_rd(ob_mol):
    """Convert OpenBabel molecule to RDKit molecule"""
    rd_mol = Chem.MolFromMol2Block(ob_mol.write("mol2"), removeHs=True)
    return rd_mol


def generate_rdmol_from_3d(atom_types: torch.Tensor, local_coords: torch.Tensor) -> Chem.rdchem.Mol:
    """
    Generate SMILES from 3D structure (single building block).
    
    Args:
        atom_types: Tensor of atomic numbers [num_atoms]
        local_coords: Tensor of 3D coordinates [num_atoms, 3]
        
    Returns:
        RDKit molecule
    """
    # Check if building block is metal
    is_metal = is_metal_bb(atom_types)
    # Create OpenBabel molecule
    ob_mol = ob_mol_from_data(atom_types, local_coords, is_metal)
    
    # Convert to RDKit molecule
    rd_mol = ob_to_rd(ob_mol)
    
    return rd_mol


def mof2string(
    block_atom_types: List[torch.Tensor],
    block_local_coords: List[torch.Tensor],
) -> str:
    """
    Convert MOF data to string.

    Args:
        block_atom_types: List of atom types for each block
        block_local_coords: List of local coordinates for each block
        
    Returns:
        sorted_indices: List of indices of the blocks in the original order
        mof_sequence: String of the MOF sequence
    """
    try:
        metal_items = []   # (orig_idx, smiles, weight)
        organic_items = [] # (orig_idx, smiles, weight)
        for orig_idx, (atom_types, coords) in enumerate(zip(block_atom_types, block_local_coords)):
            rdmol = generate_rdmol_from_3d(atom_types, coords)            
            weight = float(Descriptors.ExactMolWt(rdmol))
            # Classify as metal or organic
            if is_metal_bb(rdmol):
                smi = Chem.MolToSmiles(rdmol, canonical=True).replace(".", "")
                metal_items.append((orig_idx, smi, weight))
            else:
                smi = Chem.MolToSmiles(rdmol, canonical=True)
                organic_items.append((orig_idx, smi, weight))
        
        # Sort by weight
        metal_items.sort(key=lambda x: x[2])
        organic_items.sort(key=lambda x: x[2])
        
        # Combine: metals first, then organics
        ordered_items = metal_items + organic_items
        sorted_indices = [item[0] for item in ordered_items]
        sorted_smiles = [item[1] for item in ordered_items]

        mof_sequence = ".".join(sorted_smiles)

        return sorted_indices, mof_sequence

    except Exception as e:
        print(f"Error converting MOF data to string: {e}")
        return None