import os

import dataclasses
import random
import torch

from transformers import (
    AutoTokenizer,
)

from .models.work4_llama.modeling_work4_llama import ResNetLlamaModel

from llama_recipes.configs import fsdp_config as FSDP_CONFIG
from llama_recipes.configs import train_config as TRAIN_CONFIG

from llama_recipes.utils.config_utils import (
    update_config,
    generate_dataset_config,
    get_dataloader_kwargs,
)
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

from llama_recipes.utils.work4_train_utils import (
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
)
from llama_recipes.utils.work4_test_utils import (
    test_conditional_generation,
)
from accelerate.utils import is_xpu_available
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape} {normal_repr(self)}"
import numpy as np

def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run

def main(**kwargs):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    update_config((train_config, fsdp_config), **kwargs)
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None

    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank==0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|image_feature|>','[Img]', '[/Img]', '[MRG]', '[MRS]', '[global_node]', "<|pathology_feature|>", "<|pathology_status|>","<|pathology_cls|>"]})

    ######## Load fine-tuning data ########
    dataset_config = generate_dataset_config(train_config, kwargs)
    # Load and preprocess the dataset for training and validation

    dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Testing Set Length = {len(dataset_test)}")

    test_dl_kwargs = get_dataloader_kwargs(train_config,dataset_test, tokenizer, "test")
    # Ensure drop_last is False
    if 'drop_last' not in test_dl_kwargs:
        test_dl_kwargs['drop_last'] = False
        
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **test_dl_kwargs,
    )

    use_cache = False if train_config.enable_fsdp else None
    image_token_index = tokenizer.encode("<|image_feature|>")[-1]
    visual_language_model = ResNetLlamaModel(train_config, use_cache, tokenizer, image_token_index, kwargs, wandb_run).cuda()
    visual_language_model.from_pretrained(train_config.output_dir)  # load trained ckpt from train_config.output_dir
    print_model_size(visual_language_model, train_config, rank if train_config.enable_fsdp else 0)

    # Start the testing process

    scores = test_conditional_generation(
        visual_language_model,
        train_config,
        test_dataloader,
        local_rank if train_config.enable_fsdp else None,
        tokenizer,
        wandb_run,
    )

    return scores