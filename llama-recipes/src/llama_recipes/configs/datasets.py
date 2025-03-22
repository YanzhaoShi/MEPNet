# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

# origin
# @dataclass
# class samsum_dataset:
#     dataset: str =  "samsum_dataset"
#     train_split: str = "train"
#     test_split: str = "validation"

#modified by zcx
@dataclass
class samsum_dataset:
    dataset: str = "samsum_dataset"
    train_split: str = "train"
    test_split: str = "test"
    validation_split: str = "validation"

#added by zcx
@dataclass
class ctrg_dataset:
    dataset: str = "ctrg_dataset"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"

#added by zcx
@dataclass
class ctrg_simple_dataset:
    dataset: str = "ctrg_simple_dataset"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"

#added by YanzhaoShi
@dataclass
class ctrg_work4_dataset:
    dataset: str = "ctrg_work4_dataset"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"

@dataclass
class ctrg_work5_dataset:
    dataset: str = "ctrg_work5_dataset"
    train_split: str = "train"
    validation_split: str = "validation"
    test_split: str = "test"

@dataclass
class grammar_dataset:
    dataset: str = "grammar_dataset"
    train_split: str = "src/llama_recipes/datasets/grammar_dataset/gtrain_10k.csv" 
    test_split: str = "src/llama_recipes/datasets/grammar_dataset/grammar_validation.csv"

    
@dataclass
class alpaca_dataset:
    dataset: str = "alpaca_dataset"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "src/llama_recipes/datasets/alpaca_data.json"
    
    
@dataclass
class custom_dataset:
    dataset: str = "custom_dataset"
    file: str = "examples/custom_dataset.py"
    train_split: str = "train"
    test_split: str = "validation"