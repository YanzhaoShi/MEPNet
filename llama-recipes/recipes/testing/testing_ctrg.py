# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import os
import pandas as pd
import itertools
from llama_recipes.test_ResNet_MLP_llama import main

if __name__ == "__main__":
    # fire.Fire(main)
    seed_list = []
    version = "v5"
    grid_params = {
            'seed': [6544],
            'temperature':[0.1],
            'batch_size': [5],  # 5,6
            'dataset': ["BCT-CHR"],  # "BCT-CHR" "CTRG"
            'model': ["baseline"],
    }
    param_combinations = list(itertools.product(*grid_params.values()))
    for params in param_combinations:
        print(params)
        dir_name = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/MEPNet/output_dir/" + f"{params[3]}/baselines/baseline_{version}_{str(params[0])}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        scores = main(seed=params[0], test_batch_size=params[2], output_dir=dir_name, temperature=params[1])