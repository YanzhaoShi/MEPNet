# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass

def get_seetings():
    '''
    Configure dataset file paths and options.
    '''
    is_BCT_CHR = True  # use BCT-CHR [True, False]
    is_Chinese_prompt = 1  # use Chinese Prompt [0, 1]
    if is_BCT_CHR:
        text_json_dataset_path = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/llama_baseline/BCT-CHR_dataset/"
        # text_json_dataset_path = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/llama_baseline/BCT-CHR_dataset_debug/"
        visual_features_dir_path = "/home/bjutcv/data/Yanzhaoshi/data/syz_fc_0218_yichang2/"
        visual_att_feature_dir_path = "/home/bjutcv/data/Yanzhaoshi/data/syz_att_0218_yichang2/"
        pathological_graph_dir_path = "/home/bjutcv/data/Yanzhaoshi/data/Pathological_Graph/"
        pathological_label_path = "/home/bjutcv/data/Yanzhaoshi/MRG_chest/data/brain_data/Pathological_labels.json"
    else:
        text_json_dataset_path = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/llama_baseline/CTRG_dataset/"
        visual_features_dir_path = "/devdata/Brain_CT_Datasets/CTRG-Brain/feature/fc/"
        visual_att_feature_dir_path = "/devdata/Brain_CT_Datasets/CTRG-Brain/feature/att/"
        pathological_graph_dir_path = "/devdata/Brain_CT_Datasets/CTRG-Brain/graph/Pathological_Graph/"
        pathological_label_path = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/llama_baseline/CTRG_dataset/Pathological_labels.json"
    info_set = {
        "text_json_dataset_path": text_json_dataset_path,  # text data
        "visual_features_dir_path": visual_features_dir_path,  # global visual features
        "visual_att_feature_dir_path": visual_att_feature_dir_path,  # local visual features
        "pathological_graph_dir_path": pathological_graph_dir_path,  # medical graph for medical entities
        "pathological_label_path": pathological_label_path,  # class labels for medical entities
    }
    return info_set, is_BCT_CHR, is_Chinese_prompt

@dataclass
class train_config:
    model_name: str="/home/bjutcv/data/LLM/Meta-Llama-3-8B"
    tokenizer_name: str=None
    enable_fsdp: bool=False
    low_cpu_fsdp: bool=False
    run_validation: bool=True
    batch_size_training: int=5
    temperature: float=0.6
    batching_strategy: str="padding" #alternative: padding packing
    context_length: int=800
    gradient_accumulation_steps: int=1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int=10  # 50, 15, 10
    max_train_step: int=0
    max_eval_step: int=0
    num_workers_dataloader: int=1
    lr: float=1e-4
    weight_decay: float=0.0
    gamma: float= 0.85
    seed: int=6544
    use_fp16: bool=False
    mixed_precision: bool=True
    val_batch_size: int=30
    test_batch_size: int=30  # 50, 32, 16, 8
    dataset = "ctrg_work4_dataset"
    peft_method: str = "lora" # None, llama_adapter (Caution: llama_adapter is currently not supported with FSDP)
    use_peft: bool=True
    output_dir: str = "PATH/to/save/output_dir/"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = True
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str="PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder: str="fine-tuned" # will be used if using FSDP
    save_optimizer: bool=False # will be used if using FSDP
    use_fast_kernels: bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    use_wandb: bool = False # Enable wandb for experient tracking
    # save_metrics: bool = False # saves training metrics to a json file for later plotting
    save_metrics: str = "PATH/to/save/metrics.json"
    flop_counter: bool = False # Enable flop counter to measure model throughput, can not be used with pytorch profiler at the same time.
    flop_counter_start: int = 3 # The step to start profiling, default is 3, which means after 3 steps of warmup stage, the profiler will start to count flops.
    use_profiler: bool = False # Enable pytorch profiler, can not be used with flop counter at the same time.
    profiler_dir: str = "PATH/to/save/profiler/results" # will be used if using profiler
