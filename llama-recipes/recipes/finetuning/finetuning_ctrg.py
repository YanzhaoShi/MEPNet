# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# import sys
# sys.path.append('/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/llama_baseline/llama-recipes/src/')

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import fire
import os
import pandas as pd
import itertools
from llama_recipes.finetuning_ResNet_MLP_llama import main

if __name__ == "__main__":
    # fire.Fire(main)
    seed_list = []
    grid_params = {
            'seed': [6544],
            'batch_size': [5],
            'dataset': ["BCT-CHR"]  # "BCT-CHR" "CTRG"
    }
    param_combinations = list(itertools.product(*grid_params.values()))
    version = "v15"
    for params in param_combinations:
        print(params)
        dir_name = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/MEPNet/output_dir/" + f"{params[2]}/baselines/baseline_{version}_{str(params[0])}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        main(seed=params[0], batch_size_training=params[1], output_dir=dir_name)