# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import os
import pandas as pd
import itertools
from llama_recipes.test_work4_llama import main

if __name__ == "__main__":
    # fire.Fire(main)
    seed_list = []
    model_name = "ours"
    version = "v8"
    grid_params = {
            'seed': [6544],
            'temperature':[0.1],
            'batch_size': [5],
            'dataset': ["BCT-CHR"],  # "BCT-CHR" "CTRG"
    }
    param_combinations = list(itertools.product(*grid_params.values()))
    for params in param_combinations:
        print(params)
        dir_name = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/MEPNet/output_dir/" + f"{params[3]}/{model_name}/work_{version}_{str(params[0])}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        # Model will load ckpt from dir_name, for example "MEPNet/output_dir/BCT-CHR/ours/work_v8_6544/peft_model_lora_adapter.pth"
        scores = main(seed=params[0], test_batch_size=params[2], output_dir=dir_name, temperature=params[1])
        results = []
        results.append({
            'Hyperparameters': params,
            'B1': scores["BLEU_1"],
            'B2': scores["BLEU_2"],
            'B3': scores["BLEU_3"],
            'B4': scores["BLEU_4"],
            'M': scores["METEOR"],
            'R': scores["ROUGE_L"],
            'C': scores["CIDEr"],
            'address': dir_name
        })
        df = pd.DataFrame(results)
        file_path = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/MEPNet/output_dir/" + f"{params[3]}/test_res.xlsx"
        try:
            existing_data = pd.read_excel(file_path)
            updated_data = pd.concat([existing_data, df], ignore_index=True)
            updated_data.to_excel(file_path, index=False)
        except FileNotFoundError:
            df.to_excel(file_path, index=False)

