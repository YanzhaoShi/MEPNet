# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import fire
import os
import pandas as pd
import itertools
from llama_recipes.finetuning_work4_llama import main
if __name__ == "__main__":
    grid_params = {
            'seed': [6544],
            'batch_size': [5],
            'dataset': ["CTRG"]  # "BCT-CHR" "CTRG"
    }
    param_combinations = list(itertools.product(*grid_params.values()))
    model_name = "ours"
    version = "v15"
    for params in param_combinations:
        print(params)
        dir_name = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/MEPNet/output_dir/" + f"{params[2]}/{model_name}/work_{version}_{str(params[0])}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        scores = main(seed=params[0], batch_size_training=params[1], output_dir=dir_name)



