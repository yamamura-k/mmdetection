import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import simdjson


def gather_assigner_info(root_path: str, n_ranks: int, n_epochs: int):
    gathered_inputs = list()
    for epoch in tqdm(range(n_epochs)):
        tmp = list()
        for rank in range(n_ranks):
            assigner_info_file=f'rank_{rank}_epoch_{epoch:0>4}.json'
            with open(os.path.join(root_path, assigner_info_file), 'r') as f:
                tmp.append(json.load(f))
        gathered_inputs.append(tmp)
    return gathered_inputs

def gather_and_reduce_assigner_info(root_path: str, n_ranks: int, n_epochs: int, keys: list[str], reduce_op: str):
    n_items = len(keys)
    gathered_inputs = [list() for _ in range(n_items)]
    _reduce_op = sum if reduce_op == 'sum' else lambda x: x
    parser = simdjson.Parser()
    for epoch in tqdm(range(n_epochs)):
        tmp = [list() for _ in range(n_items)]
        for rank in range(n_ranks):
            assigner_info_file=f'rank_{rank}_epoch_{epoch:0>4}.json'
            # with open(os.path.join(root_path, assigner_info_file), 'r') as f:
            #     tmp_data = json.load(f)
            with open(
                os.path.join(root_path, assigner_info_file), 
                'r', encoding="utf-8"
                ) as f:
                tmp_data = parser.parse(f.read())
            for _i, key in enumerate(keys):
                tmp_value = _reduce_op(map(_reduce_op, tmp_data[key]))
                tmp[_i].append(tmp_value)
            del tmp_data
        for _j in range(n_items):
            gathered_inputs[_j].append(_reduce_op(tmp[_j]))
    return gathered_inputs

def summarize_assigner_info(gathered_inputs):
    dynamic_ks = list(map(lambda x: sum(map(lambda y: sum(map(sum, y['dynamic_ks'])), x)), tqdm(gathered_inputs)))
    num_assigned = list(map(lambda x: sum(map(lambda y: sum(map(sum, y['num_assigned_per_gt'])), x)), tqdm(gathered_inputs)))
    return dynamic_ks, num_assigned

def plot_diff_assign():
    WORK_DIRS='work_dirs'
    MODEL_NAMES=['yolox_s_8xb8-300e_coco', 'yolox_s_ota_8xb8-300e_coco', 'yolox_s_uota_8xb8-300e_coco']
    CKPT_DATE=[
        '20250327_181701',
        '20250331_132142',
        '20250405_232725',
    ]
    ASSIGNER_INFO='assigner_info'
    N_MODELS=len(MODEL_NAMES)
    N_RANKS=4
    N_EPOCHS=170
    KEYS = ['dynamic_ks', 'num_assigned_per_gt']
    REDUCE_OP='sum'
    data = [list(), list()]
    for i in range(N_MODELS):
        _root_dir = os.path.join(WORK_DIRS, MODEL_NAMES[i], CKPT_DATE[i], ASSIGNER_INFO)
        dynamic_ks, num_assigned = gather_and_reduce_assigner_info(_root_dir, N_RANKS, N_EPOCHS, KEYS, REDUCE_OP)
        dk_array = np.asarray(dynamic_ks)
        nass_array = np.asarray(num_assigned)
        diff = nass_array - dk_array
        data[0].append(diff)
        data[1].append(diff / dk_array)
    fig, ax = plt.subplots(1, 2, )
    for i in range(N_MODELS):
        ax[0].plot(data[0][i], label=MODEL_NAMES[i], c=f"C{i}")
        ax[1].plot(data[1][i], label=MODEL_NAMES[i], c=f"C{i}")
    ax[0].legend()
    ax[0].set_title('#assigned - dynamic_ks')
    ax[1].legend()
    ax[1].set_title('(#assigned - dynamic_ks) / #dynamic_ks')
    fig.tight_layout()
    fig.savefig('test_plot.png', dpi=300)
    plt.clf();plt.close()
    
if __name__=='__main__':
    plot_diff_assign()
    
    