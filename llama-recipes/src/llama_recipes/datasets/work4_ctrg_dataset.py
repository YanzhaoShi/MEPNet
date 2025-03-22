import copy
import json

import torch
import numpy as np
import datasets
from ..configs.training import get_seetings

def get_preprocessed_ctrg(dataset_config, tokenizer, split):

    info_set, is_BCT_CHR, is_Chinese_prompt =  get_seetings()
    text_json_dataset_path = info_set["text_json_dataset_path"]
    visual_features_dir_path = info_set["visual_features_dir_path"]
    pathological_graph_dir_path = info_set["pathological_graph_dir_path"]
    pathological_label_path = info_set["pathological_label_path"]

    pathology_entity_Chinese = ['侧脑室', '脑干', '上颌窦', '第三脑室', '顶叶', '脑室',
                                     '半卵圆中心', '枕叶',
                                     '中线', '第四脑室',
                                     '脑沟', '颞枕叶', '额叶', '基底节区', '脑实质', '筛窦', '颞叶', '蝶窦', '丘脑',
                                     '高密度影', '低密度影',
                                     '减低', "受压", '水肿带', '增宽', '增厚', '增高', '密度影', '变窄', '左移',
                                     '移位', '肿胀']
    pathology_entity_Chinese_half = ['侧脑室', '脑干', '上颌窦', '第三脑室', '顶叶', '脑室',
                                     '半卵圆中心', '枕叶',
                                     '中线',
                                     '高密度影', '低密度影',
                                     '减低', "受压", '水肿带', '增宽', '增厚']
    entity_num = len(pathology_entity_Chinese)
    entity_num_half = len(pathology_entity_Chinese_half)

    dataset = datasets.load_dataset(text_json_dataset_path, split=split)

    if is_Chinese_prompt == 1:
        # Use Chinese Prompt
        visual_prompt = "<|image_feature|>" * 24
        model_settings = "ours"
        if model_settings == "ours":
            pathology_prompt = f"另外，可参考以下{entity_num}种病理实体的视觉编码和学习状态："
            for idx, entity in enumerate(pathology_entity_Chinese):
                if idx == entity_num-1:
                    pathology_prompt = pathology_prompt + f"{entity}<|pathology_feature|><|pathology_status|>。"
                else:
                    pathology_prompt = pathology_prompt + f"{entity}<|pathology_feature|><|pathology_status|>;"
            prompt_MRG = (
                f"[Img]{{visual_prompt}}[/Img][MRG]详细地用中文描述多张脑CT影像并生成脑CT报告。{{pathology_prompt}}"
            )

    else:
        # Use English Prompt
        visual_prompt = "<|image_feature|>" * 24
        pathology_prompt = f"You should refer to the embedding and learning statuses of the following {entity_num} medical terms:"
        for idx, entity in enumerate(pathology_entity_Chinese):
            if idx == entity_num - 1:
                pathology_prompt = pathology_prompt + f"{entity}<|pathology_feature|><|pathology_status|>。"
            else:
                pathology_prompt = pathology_prompt + f"{entity}<|pathology_feature|><|pathology_status|>;"

        prompt_MRG = (
            f"[Img]{{visual_prompt}}[/Img][MRG]Describe multiple brain CT images in detail and generate a Chinese brain CT report. {{pathology_prompt}}"
        )
    
    def apply_prompt_template(sample):
        return {
            "id": sample["id"],
            "prompt": prompt_MRG.format(visual_prompt=visual_prompt, pathology_prompt=pathology_prompt),
            "findings": sample["findings"],
            "impression": sample["impression"],
        }

    dataset = dataset.map(apply_prompt_template, remove_columns=list(dataset.features))

    def tokenize_add_label(sample):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        findings = tokenizer.encode(sample["findings"] + tokenizer.eos_token, add_special_tokens=False)
        
        image_features_path = visual_features_dir_path + str(sample["id"]) + ".npy"
        pathological_graph_path = pathological_graph_dir_path + str(sample["id"]) + ".pt"
        image_features = torch.from_numpy(np.load(image_features_path))
            
        pathological_graph = torch.load(pathological_graph_path)
        pathological_label = torch.tensor(json.loads(open(pathological_label_path, 'r').read())[str(sample["id"])])

        if split == "train":
            sample = {
                "input_ids": prompt + findings,
                "attention_mask" : [1] * (len(prompt) + len(findings)),
                "labels": [-100] * len(prompt) + findings,
                "image_features": image_features,
                "image_id": int(sample["id"]),
                "pathological_graph": pathological_graph,
                "pathological_label": pathological_label,
            }
        else:
            sample = {
            "id": [int(sample["id"])],
            "input_ids": prompt,
            "attention_mask" : [1] * len(prompt),
            "labels": findings,
            "image_features": image_features,
            "image_id": int(sample["id"]),
            "pathological_graph": pathological_graph,
            "pathological_label": pathological_label,
            }


        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
