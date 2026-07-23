# MOFFlow Prompt Conversion

Scripts for converting MOFFlow processed data (`*_processed.pkl.gz`) into the Alpaca-style SFT prompt JSON files used by MOF-LLM.


## Files

- `extract_bb_smiles.py`: extracts ordered building-block SMILES from `*_processed.pkl.gz`.
- `build_sft_prompts_final.py`: builds `sft-{split}.json` prompts from processed data and extracted SMILES.
- `rebuild_mofflow_prompts_final.sh`: runs both stages for `train`, `val`, and `test`.

## Input

```text
{split}_processed.pkl.gz
```

## Outputs

Intermediate SMILES generated from the pkl files:

```text
{split}_bb_smiles.json
```

Final SFT prompts:

```text
sft-{split}.json
```

## Conversion Logic

Stage 1 reads `atom_types`, `local_coords`, and `bb_num_vec` to reconstruct each building block and write canonical SMILES in the original building-block order.

Stage 2 writes each prompt as:

- `instruction`: MOF-LLM structure-prediction instruction.
- `input`: ordered building-block SMILES.
- `output`: lattice, fractional translations, and Euler-angle rotations.

Field mapping:

- lattice: `lattice_1`, formatted to 2 decimals.
- translation: `trans_1 @ inverse(cell_1)`, formatted to 3 decimals.
- rotation: `rotmats_1` converted to Euler angles in radians, formatted to 3 decimals.

Records with more than 200 building blocks or failed SMILES extraction are skipped.

## Usage

Run from the repository root:

```bash
bash data_processing/build_mofflow_prompts.sh
```

To rebuild only one split:

```bash
python data_processing/extract_bb_smiles.py \
  --splits test \
  --data-dir data/mofflow \
  --output-dir data/mofflow/prompts_rebuild/bb_smiles

python data_processing/build_sft_prompts_final.py \
  --splits test \
  --data-dir data/mofflow \
  --smiles-dir data/mofflow/prompts_rebuild/bb_smiles \
  --output-dir data/mofflow/prompts_final
```

Arguments used in the commands above:

- `--data-dir`: directory containing the MOFFlow processed input files, such as
  `{split}_processed.pkl.gz`. For example, `data/mofflow` means the scripts will
  read `data/mofflow/test_processed.pkl.gz` when `--splits test` is used.
- `--output-dir` in `extract_bb_smiles.py`: directory where extracted
  building-block SMILES files are written, such as `{split}_bb_smiles.json`.
  This directory is usually used as `--smiles-dir` in the next stage.
- `--smiles-dir`: directory containing the extracted building-block SMILES files
  produced by `extract_bb_smiles.py`, such as `{split}_bb_smiles.json`.
- `--output-dir` in `build_sft_prompts_final.py`: directory where the final
  Alpaca-style SFT prompt files are written, such as `sft-{split}.json`.
