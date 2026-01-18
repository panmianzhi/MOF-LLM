from collections import defaultdict
import json
import pickle
from functools import partial
import gzip
import json
import numpy as np
import torch
import tqdm
from p_tqdm import p_map
from typing import Dict, List
from pymatgen.core import Structure, Lattice, Element
from pymatgen.analysis.structure_matcher import StructureMatcher
import matplotlib.pyplot as plt
import fire

from common.tokenize_rotation import rotation_vector_to_axis_angle, \
    axis_angle_to_rotation_matrix, assemble_coords
from common.rotation_euler import euler_to_matrix



def prompt_to_structure(
    prompt: str,
    data_dict: Dict,
    use_euler: bool = True,
) -> Structure:
    """
    transform prompt to structure where blocks order is the same as the original data
    Args:
        prompt: "a b c alpha beta gamma\n[0] tx ty tz rx ry rz\n[1] tx ty tz rx ry rz\n..."
        data_dict: structure data loaded from pkl.gz file
        the i-th block in prompt is the sorted_bb_indices[i]-th block in data_dict
    """
    clean_prompt = prompt.replace('<think>\n\n</think>\n\n', '').strip().split('\n') # first is lattice, then block translations and rotations

    pred_lattice = clean_prompt.pop(0) # "a b c alpha beta gamma"
    pred_lattice_params = []
    for l_id, p in enumerate(pred_lattice.split(' ')):
        param = float(p)
        if param <= 0: 
            raise ValueError(f"Lattice parameter {p} is negative at index {l_id}")
        elif l_id > 2 and param < 1:
            raise ValueError(f"Lattice angle {p} is less than 1 at index {l_id}")
        pred_lattice_params.append(param)
    pred_lattice = Lattice.from_parameters(*pred_lattice_params)
    pred_cell = torch.tensor(pred_lattice.matrix, dtype=torch.float32)

    if len(clean_prompt) > data_dict['bb_num_vec'].shape[0]:
        clean_prompt = clean_prompt[:data_dict['bb_num_vec'].shape[0]]
    elif len(clean_prompt) < data_dict['bb_num_vec'].shape[0]:
        raise ValueError(f"#blocks in prompt ({len(clean_prompt)}) < #blocks in data_dict ({data_dict['bb_num_vec'].shape[0]})")

    trans_frac_all = []
    rotmats_all = []
    for block_str in clean_prompt:
        block_str = block_str.strip().split(' ')
        block_str = [float(p) for p in block_str[1:]] # remove [block_id]
        
        trans_frac = torch.tensor(block_str[:3], dtype=torch.float32)
        
        rotvec = torch.tensor(block_str[3:], dtype=torch.float32)
        if use_euler:
            rotmat = euler_to_matrix(rotvec, degrees=False) # (3,3)
        else:
            axis_angle = rotation_vector_to_axis_angle(rotvec) # (axis, angle)
            rotmat = axis_angle_to_rotation_matrix(*axis_angle) # (3,3)
        
        trans_frac_all.append(trans_frac)
        rotmats_all.append(rotmat)

    trans_frac_all = torch.stack(trans_frac_all) # (num_bbs, 3)
    trans_cart_all = torch.matmul(trans_frac_all, pred_cell) # (num_bbs, 3)
    rotmats_all = torch.stack(rotmats_all) # (num_bbs, 3, 3)

    local_coords = data_dict['local_coords']
    pred_global_coords = assemble_coords(local_coords, rotmats_all, trans_cart_all, data_dict['bb_num_vec']) # (num_atoms, 3)

    structure = Structure(
        lattice=pred_lattice,
        species=[Element.from_Z(z).symbol for z in data_dict['atom_types'].tolist()],
        coords=pred_global_coords,
        coords_are_cartesian=True,
    )

    return structure

def test_prompt_to_structure():
    """
    test whether prompt_to_structure is correct
    """
    gt_mof_struct = 'data/mofflow/val_processed.pkl.gz'
    prompt_file = '...'

    with open(prompt_file, 'r') as f:
        prompt_data = json.load(f)

    struct_data = pickle.load(gzip.open(gt_mof_struct, 'rb'))

    label_structs = []
    gt_structs = []
    for i, prompt in tqdm.tqdm(enumerate(prompt_data)):
        label = prompt['output']
        label_struct = prompt_to_structure(label, struct_data[i], use_euler=True)
        gt_struct = Structure(
            lattice=Lattice(struct_data[i]['cell_1']),
            species=[Element.from_Z(z).symbol for z in struct_data[i]['atom_types'].tolist()],
            coords=struct_data[i]['gt_coords'],
            coords_are_cartesian=True,
        )
        label_structs.append(label_struct)
        gt_structs.append(gt_struct)
        if i == 1000: break

    matcher = StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10) # Default values
    results = defaultdict(list)
    p_results = p_map(
        partial(calculate_rms_dist, matcher=matcher),
        label_structs,
        gt_structs,
        num_cpus=20,
    )

    results["rms_dist"] = p_results
    results["match_rate"] = sum(rms is not None for rms in results["rms_dist"]) / len(results["rms_dist"]) * 100

    # Print results
    print("Average RMSD:", np.mean([rms for rms in results["rms_dist"] if rms is not None]))
    print("Match rate (%):", results["match_rate"])


def calculate_rms_dist(
    pred_struct: Structure,
    gt_struct: Structure,
    matcher: StructureMatcher,
):
    try:
        rms_dist = matcher.get_rms_dist(gt_struct, pred_struct)
        rms_dist = None if rms_dist is None else rms_dist[0]
    except Exception as e:
        print(f"Error: {e}")
        rms_dist = None
    return rms_dist

def main(
    pred_jsonl_file: str,
    gt_pkl_file: str = "data/mofflow/test_processed.pkl.gz",
    stol: float = 0.5,
    num_cpus: int = 20,
):
    with gzip.open(gt_pkl_file, 'rb') as f:
        gt_data = pickle.load(f)

    matcher = StructureMatcher(stol=stol, ltol=0.3, angle_tol=10) # Default values
    results = defaultdict(list)

    pred_structs = []
    gt_structs = []
    with open(pred_jsonl_file, 'r') as f:
        for mof_id, line in tqdm.tqdm(enumerate(f)):
            pred_data = json.loads(line)
            try:
                gt_mof = gt_data[mof_id]
                pred_struct = prompt_to_structure(pred_data['predict'], gt_mof, use_euler=True) # sorted_bb_indices[str(mof_id)]
                gt_struct = Structure(
                    lattice=Lattice(gt_data[mof_id]['cell_1']),
                    species=[Element.from_Z(z).symbol for z in gt_mof['atom_types'].tolist()],
                    coords=gt_mof['gt_coords'],
                    coords_are_cartesian=True,
                )
                pred_structs.append(pred_struct)
                gt_structs.append(gt_struct)
            except Exception as e:
                print(f"Skip mof {mof_id}: {e}")
                continue


    p_results = p_map(
        partial(calculate_rms_dist, matcher=matcher),
        pred_structs,
        gt_structs,
        num_cpus=num_cpus,
    )

    # save p_results to json
    save_name = pred_jsonl_file.split('/')[-1].split('.')[0] + '_stol_{}_res.json'.format(stol)
    with open(save_name, 'w') as f:
        json.dump(p_results, f)

    results["rms_dist"] = p_results
    results["match_rate"] = sum(rms is not None for rms in results["rms_dist"]) / len(results["rms_dist"]) * 100

    # Print results
    print("Average RMSD:", np.mean([rms for rms in results["rms_dist"] if rms is not None]))
    print("Match rate (%):", results["match_rate"])


if __name__ == '__main__':
    fire.Fire(main)
    #test_prompt_to_structure()