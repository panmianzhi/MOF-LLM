#!/usr/bin/env bash
set -euo pipefail

SMILES_DIR=data/mofflow/prompts_rebuild/bb_smiles
OUTPUT_DIR=data/mofflow/prompts_final

python prompts_final_rebuild/extract_bb_smiles.py   --splits train val test   --data-dir data/mofflow   --output-dir "$SMILES_DIR"

python prompts_final_rebuild/build_sft_prompts_final.py   --splits train val test   --data-dir data/mofflow   --smiles-dir "$SMILES_DIR"   --output-dir "$OUTPUT_DIR"
