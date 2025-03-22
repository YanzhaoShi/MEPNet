

# MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation

This is the official code of "MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation" (AAAI 2025 oral)


## Abstract
The automatic generation of brain CT reports has gained widespread attention, given its potential to assist radiologists in diagnosing cranial diseases. However, brain CT scans involve extensive medical entities, such as diverse anatomy regions and lesions, exhibiting highly inconsistent spatial patterns in 3D volumetric space. This leads to biased learning of medical entities in existing methods, resulting in repetitiveness and inaccuracy in generated reports. To this end, we propose a Medical Entity-balanced Prompting Network (MEPNet), which harnesses the large language model (LLM) to fairly interpret various entities for accurate brain CT report generation. By introducing the visual embedding and the learning status of medical entities as enriched clues, our method prompts the LLM to balance the learning of diverse entities, thereby enhancing reports with comprehensive findings. First, to extract visual embedding of entities, we propose Knowledge-driven Joint Attention to explore and distill entity patterns using both explicit and implicit medical knowledge. Then, a Learning Status Scorer is designed to evaluate the learning of entity visual embeddings, resulting in unique learning status for individual entities. Finally, these entity visual embeddings and status are elaborately integrated into multi-modal prompts, to guide the text generation of LLM. This process allows LLM to self-adapt the learning process for biased-fitted entities, thereby covering detailed findings in generated reports. We conduct experiments on two brain CT report generation benchmarks, showing the effectiveness in clinical accuracy and text coherence.


## Environment
Our implementation is based on [Llama-recipes](https://github.com/meta-llama/llama-cookbook). Please refer to their repository for detailed environment setup instructions. Alternatively, you can set up the environment using the following commands:

```bash
pip install -r requirements.txt
```

Then run the following command:
```bash
cd llama-recipes
pip install -U pip setuptools
pip install -e .
```

## Data
Our experiments are mainly conducted on the BCT-CHR dataset. However, due to privacy restrictions, access to this dataset is limited.

Additionally, we evaluate our model on the publicly available [CTRG-Brain](https://github.com/tangyuhao2016/CTRG) dataset to further validate its performance. You can download this dataset from its official repository.


## Model Preparation  
To train **MEPNet**, you need to prepare some pretrained weights to enhance performance.  

### Vision Encoder  
We use **ResNet101** to extract brain CT image features. To improve training efficiency, this process is **precomputed**, meaning that the visual encoder’s parameters remain **frozen** during fine-tuning.  

Alternatively, you can explore other powerful vision encoders, such as:  
- [**CLIP**](https://github.com/openai/CLIP)  
- [**BiomedCLIP**](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)  

For feature extraction, you can refer to the script from [PCRL-MRG](https://github.com/Chauncey-Jheng/PCRL-MRG/blob/main/data_peparation/get_visual_feature/resnet101_2048.py).  

### Language Model  
We use **LLaMA 3-8B** as the language model. You can follow the instructions in **Llama-recipes** to set up and configure the model accordingly.  


## Training  
If you want to modify the model configuration or adapt it to your own dataset, you can update the following file:  

```bash
MEPNet/llama-recipes/src/llama_recipes/configs/training.py
```  

After making the necessary changes, run the following command to start training:  

```bash
cd llama-recipes
python recipes/finetuning/finetuning_work4.py
```  

## Evaluation  
To evaluate the model, run:  

```bash
cd llama-recipes
python recipes/testing/testing_work4.py
```  


## Citations
If this project is helpful to you, please consider citing:

```
@inproceedings{Zhang2025MEPNet,
  author       = {Xiaodan Zhang and
                  Yanzhao Shi and
                  Junzhong Ji and
                  Chengxin Zheng and
                  Liangqiong Qu},
  title        = {MEPNet: Medical Entity-balanced Prompting Network for Brain CT Report Generation},
  booktitle    = {AAAI Conference on Artificial Intelligence},
  year         = {2025}
}

@inproceedings{Zheng2024See,
  author       = {Chengxin Zheng and
                  Junzhong Ji and
                  Yanzhao Shi and
                  Xiaodan Zhang and
                  Liangqiong Qu},
  title        = {See Detail Say Clear: Towards Brain {CT} Report Generation via Pathological
                  Clue-driven Representation Learning},
  booktitle    = {Findings of the Association for Computational Linguistics: {EMNLP}
                  2024, Miami, Florida, USA, November 12-16, 2024},
  pages        = {16542--16552},
  publisher    = {Association for Computational Linguistics},
  year         = {2024}
}
```

## Acknowledgment
[Llama-recipes](https://github.com/meta-llama/llama-cookbook)

[PCRL-MRG](https://github.com/Chauncey-Jheng/PCRL-MRG/blob/main/data_peparation/get_visual_feature/resnet101_2048.py)











