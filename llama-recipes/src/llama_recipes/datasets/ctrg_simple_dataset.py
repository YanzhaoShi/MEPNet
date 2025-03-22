import copy
import torch
import numpy as np
import datasets



def get_preprocessed_ctrg(dataset_config, tokenizer, split):

    text_json_dataset_path = "/home/bjutcv/data/zcx/llama3/CTRG_dataset_simple_for_test/"
    visual_features_dir_path = "/home/bjutcv/data/zcx/dataset/CTRG-Brain/feature/"

    dataset = datasets.load_dataset(text_json_dataset_path, split=split)

    visual_prompt = "<|image_feature|>" * 24

    # prompt_MRG = (
    #     f"[Img]{{visual_prompt}}[/Img][MRG]详细地用中文描述给定的多张脑CT图片并生成一份中文的脑CT报告。"
    # )

    prompt_MRG = (
        f"[Img]{{visual_prompt}}[/Img][MRG]Describe multiple brain CT images in detail and generate a Chinese brain CT report."
    )

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
