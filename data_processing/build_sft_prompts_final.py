#!/usr/bin/env python
"""Build MOFFlow SFT prompt JSON files from processed pickle data."""

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import torch
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.rotation_euler import matrix_to_euler
from common.tokenize_rotation import isRotationMatrix


FINAL_INSTRUCTION = (
    "You are tasked with predicting the 3D crystal structure of a Metal-Organic "
    "Framework (MOF). MOFs are highly ordered porous materials formed by "
    "connecting metal-containing nodes with organic linkers, creating modular "
    "structures. \n"
    "Given the SMILES of all MOF building blocks, predict the complete 3D "
    "crystal structure configuration including:\n"
    "Lattice parameters\u200b in the format 'a b c α β γ' "
    "(where a, b, c are unit cell lengths, and α, β, γ are unit cell angles). \n"
    "For each building block\u200b (maintaining the exact same order as provided in the input):\n"
    "Translation vector\u200b ([tx ty tz]): The position of the building block's center "
    "within the unit cell, expressed in fractional coordinates.\n"
    "Rotation angles\u200b ([roll pitch yaw]): The orientation of the building block, "
    "represented in radians\u200b using Euler angles.\n"
    "Output Format:\n"
    "First line: Lattice parameters, i.e.: a b c α β γ.\n"
    "Subsequent lines (one per building block): [k] tx ty tz roll pitch yaw\n"
    "k: The 0-based index of the building block.\n"
    "tx ty tz: Fractional coordinates within the unit cell.\n"
    "roll pitch yaw: Euler angles in radians.\n"
)

INPUT_PREFIX = "Input Building Blocks(Separate by spaces): "


def build_sft_prompts(
    split: str,
    data_dir: Path,
    smiles_dir: Path,
    output_dir: Path,
    use_fractional_translation: bool = True,
) -> Path:
    """Build sft-{split}.json from processed MOF data and extracted SMILES."""
    smiles_path = smiles_dir / f"{split}_bb_smiles.json"
    data_path = data_dir / f"{split}_processed.pkl.gz"
    output_path = output_dir / f"sft-{split}.json"

    with open(smiles_path) as f:
        bb_smiles = json.load(f)

    with gzip.open(data_path, "rb") as f:
        mof_data = pickle.load(f)

    all_prompts = []
    for mof_id, mof in tqdm.tqdm(
        enumerate(mof_data),
        total=len(mof_data),
        desc=f"build {split} prompts",
    ):
        mof_key = str(mof_id)
        rotmats = mof["rotmats_1"]
        num_bbs = rotmats.shape[0]
        if num_bbs > 200 or mof_key not in bb_smiles:
            print(f"invalid mof {mof_id}!")
            continue

        smiles_list = bb_smiles[mof_key]
        if len(smiles_list) != num_bbs:
            raise ValueError(
                f"{split} MOF {mof_id}: {len(smiles_list)} SMILES for {num_bbs} BBs"
            )

        trans = mof["trans_1"]
        if use_fractional_translation:
            trans_all = torch.matmul(trans, torch.linalg.inv(mof["cell_1"]))
        else:
            trans_all = trans

        lattice_str = " ".join(f"{x:.2f}" for x in mof["lattice_1"].tolist())
        output_lines = [lattice_str]
        for bb_idx in range(num_bbs):
            rotmat = rotmats[bb_idx]
            if isRotationMatrix(rotmat):
                euler = matrix_to_euler(rotmat, degrees=False).tolist()
            else:
                print(f"Warning: invalid rotation matrix for MOF {mof_id}, BB {bb_idx}")
                euler = [0.0, 0.0, 0.0]

            trans_str = " ".join(f"{x:.3f}" for x in trans_all[bb_idx].tolist())
            euler_str = " ".join(f"{x:.3f}" for x in euler)
            output_lines.append(f"[{bb_idx}] {trans_str} {euler_str}")

        all_prompts.append(
            {
                "instruction": FINAL_INSTRUCTION,
                "input": INPUT_PREFIX + " ".join(smiles_list),
                "output": "\n".join(output_lines),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_prompts, f, indent=4, ensure_ascii=False)

    print(f"Wrote {len(all_prompts)} prompts to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: build final SFT prompts from processed data and BB SMILES."
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/mofflow"))
    parser.add_argument(
        "--smiles-dir",
        type=Path,
        default=Path("data/mofflow/prompts_rebuild/bb_smiles"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mofflow/prompts_final"),
    )
    parser.add_argument(
        "--cartesian-translation",
        action="store_true",
        help="Use Cartesian translations. The final prompts use fractional translations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for split in args.splits:
        build_sft_prompts(
            split=split,
            data_dir=args.data_dir,
            smiles_dir=args.smiles_dir,
            output_dir=args.output_dir,
            use_fractional_translation=not args.cartesian_translation,
        )


if __name__ == "__main__":
    main()
