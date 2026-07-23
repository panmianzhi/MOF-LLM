#!/usr/bin/env python
"""Extract per-building-block SMILES from MOFFlow processed pickle files."""

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import torch
import tqdm
from rdkit import Chem, RDLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.utils import generate_rdmol_from_3d, is_metal_bb

RDLogger.DisableLog("rdApp.*")


def generate_mof_sequence(
    atom_types: torch.Tensor,
    atom_coords: torch.Tensor,
    bb_num_vec: torch.Tensor,
    remove_metal_dot: bool = False,
) -> list[str]:
    """Generate ordered canonical SMILES for each MOF building block."""
    atom_type_blocks = torch.split(atom_types, bb_num_vec.tolist())
    coord_blocks = torch.split(atom_coords, bb_num_vec.tolist())

    all_smiles = []
    for bb_atom_types, bb_coords in zip(atom_type_blocks, coord_blocks):
        mol = generate_rdmol_from_3d(bb_atom_types, bb_coords)
        smiles = Chem.MolToSmiles(mol, canonical=True)
        if remove_metal_dot and is_metal_bb(mol):
            smiles = smiles.replace(".", "")
        all_smiles.append(smiles)

    return all_smiles


def extract_bb_smiles(
    split: str,
    data_dir: Path,
    output_dir: Path,
    remove_metal_dot: bool = True,
) -> Path:
    """Write ``{split}_bb_smiles.json`` in the same BB order as ``bb_num_vec``."""
    input_path = data_dir / f"{split}_processed.pkl.gz"
    output_path = output_dir / f"{split}_bb_smiles.json"

    with gzip.open(input_path, "rb") as f:
        processed_data = pickle.load(f)

    bb_smiles = {}
    for data_idx, mof in tqdm.tqdm(
        enumerate(processed_data),
        total=len(processed_data),
        desc=f"extract {split} smiles",
    ):
        atom_types = mof["atom_types"]
        local_coords = mof["local_coords"]
        bb_num_vec = mof["bb_num_vec"]
        assert len(atom_types) == int(torch.sum(bb_num_vec))

        try:
            smiles = generate_mof_sequence(
                atom_types,
                local_coords,
                bb_num_vec,
                remove_metal_dot=remove_metal_dot,
            )
        except Exception as exc:
            print(f"Error in generating molecule for {data_idx}: {exc}")
            continue

        bb_smiles[str(data_idx)] = smiles

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(bb_smiles, f)

    print(f"Wrote {len(bb_smiles)} records to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: extract building-block SMILES from {split}_processed.pkl.gz."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mofflow"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mofflow/prompts_rebuild/bb_smiles"),
    )
    parser.add_argument(
        "--keep-metal-dot",
        action="store_true",
        help="Keep disconnected metal-cluster dots. The final prompts use dots removed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        extract_bb_smiles(
            split=split,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            remove_metal_dot=not args.keep_metal_dot,
        )


if __name__ == "__main__":
    main()
