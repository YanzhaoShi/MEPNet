import copy
import torch
import numpy as np
import datasets



def get_preprocessed_ctrg(dataset_config, tokenizer, split):

    is_BCT_CHR = True  # True False
    if is_BCT_CHR:
        text_json_dataset_path = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/llama_baseline/BCT-CHR_dataset/"
        visual_features_dir_path = "/home/bjutcv/data/Yanzhaoshi/data/syz_fc_0218_yichang2/"
    else:
        text_json_dataset_path = "/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/llama_baseline/CTRG_dataset/"
        visual_features_dir_path = "/devdata/Brain_CT_Datasets/CTRG-Brain/feature/"



    dataset = datasets.load_dataset(text_json_dataset_path, split=split)

    visual_prompt = "<|image_feature|>" * 24

    '''
    Prompt modification requires changing the injection position of the corresponding visual features in the visual fusion layer, in the file modelling_llama.py, within the LlamaModel method.
    '''

    # Chinese Prompt
    prompt_MRG = (
        f"[Img]{{visual_prompt}}[/Img][MRG]详细地用中文描述给定的多张脑CT图片并生成一份中文的脑CT报告。"
    )

    # English Prompt
    # prompt_MRG = (
    #     f"[Img]{{visual_prompt}}[/Img][MRG]Describe multiple brain CT images in detail and generate a Chinese brain CT report."
    # )

    def apply_prompt_template(sample):
        return {
            "id": sample["id"],
            "prompt": prompt_MRG.format(visual_prompt=visual_prompt),
            "findings": sample["findings"],
            "impression": sample["impression"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        findings = tokenizer.encode(sample["findings"] + tokenizer.eos_token, add_special_tokens=False)
        if is_BCT_CHR:
            image_features_path = visual_features_dir_path + str(sample["id"]) + ".npy"
        else:
            image_features_path = visual_features_dir_path + "fc/" + str(sample["id"]) +".npy"
        image_features = torch.from_numpy(np.load(image_features_path))

        if split == "train":
            sample = {
                "input_ids": prompt + findings,
                "attention_mask" : [1] * (len(prompt) + len(findings)),
                "labels": [-100] * len(prompt) + findings,
                "image_features": image_features,
            }
        else:
            sample = {
            "id": [int(sample["id"])],
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "labels": findings,
            "image_features": image_features,
            }


        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
