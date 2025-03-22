# coding:utf-8
import numpy as np
import itertools
import matplotlib.pyplot as plt
import torch
import json


# Construct brain tissue relationships
# More details see EMNLP 2023 Paper "Granularity Matters: Pathological Graph-driven Cross-modal Alignment for Brain CT Report Generation"
tissue_base = {'颅底层面眦耳线层面': ['上颌窦', '筛窦', '蝶窦', '第四脑室', '脑干', '小脑', '小脑半球', '桥脑'],
    '鞍上池层面': ['额叶', '颞叶', '颞枕叶', '枕叶', '中脑'],
    '第三脑室下部层面': ['额叶', '丘脑', '侧脑室', '枕叶', '第三脑室', '小脑幕', '颞叶', '中线'],
    '第三脑室上部层面': ['基底节区', '大脑镰', '侧脑室', '枕叶', '颞枕叶', '丘脑', '颞叶', '第三脑室', '额叶', '中线'],
    '侧脑室体部层面': ['侧脑室', '额叶', '放射冠', '颞叶', '枕叶', '颞枕叶', '大脑镰', '中线'],
    '侧脑室上部层面': ['侧脑室', '大脑镰', '额叶', '顶叶', '脑沟', '脑室', '枕叶', '中线'],
    '半卵圆中心层面': ['额叶', '半卵圆中心', '顶叶', '大脑镰', '脑沟', '脑室', '脑实质', '中线'],
    '大脑皮质上部层面': ['大脑镰', '脑回', '顶叶', '放射冠', '额叶', '脑沟']}
tissue_words=['侧脑室', '脑干', '上颌窦', '第三脑室', '顶叶', '脑室', '半卵圆中心', '枕叶', '中线', '第四脑室', '脑沟', '颞枕叶',
        '额叶', '基底节区', '脑实质', '筛窦', '颞叶', '蝶窦', '丘脑']
tissue_dict={
    '侧脑室':1,
    '脑干':2,
    '上颌窦':3,
    '第三脑室':4,
    '顶叶':5,
    '脑室':6,
    '半卵圆中心':7,
    '枕叶':8,
    '中线':9,
    '第四脑室':10,
    '脑沟':11,
    '颞枕叶':12,
    '额叶':13,
    '基底节区':14,
    '脑实质':15,
    '筛窦':16,
    '颞叶':17,
    '蝶窦':18,
    '丘脑':19,
}
tissue_number = len(tissue_words)
print("A total of {} tissue keywords are included.".format(tissue_number))
Tissue_A = np.eye(tissue_number,k=0)

layer_tissue_number = 0
tissue_tissue_number = 0
# Traverse the levels
for scan in tissue_base:
    # Organization at the current level
    scan_relation_base=tissue_base[scan]
    layer_tissue_number += len(scan_relation_base)
    # print(scan_relation_base)
    # print(len(scan_relation_base))
    scan_relation_base_index=[]
    for x in scan_relation_base:
        try:
            scan_relation_base_index.append(tissue_dict[x])
        except:
            pass
    # The current `scan_relation_base_index` has obtained the key parts at this level. Next, pair them to find relationships.
    # print(scan_relation_base_index) [13, 17, 12, 8]
    scan_relation_pair= list(itertools.permutations(scan_relation_base_index, 2))
    # print(scan_relation_pair) [(13, 17), (13, 12), (13, 8), (17, 13), (17, 12), (17, 8), (12, 13), (12, 17), (12, 8), (8, 13), (8, 17), (8, 12)]
    # print(len(scan_relation_pair))
    tissue_tissue_number += len(scan_relation_pair)
    for relation in scan_relation_pair:
        if(Tissue_A[int(relation[0]-1)][int(relation[1]-1)]<1):
            Tissue_A[int(relation[0]-1)][int(relation[1]-1)]+=0.1

# Construct brain lesion relationships
# More details see EMNLP 2023 Paper "Granularity Matters: Pathological Graph-driven Cross-modal Alignment for Brain CT Report Generation"
lesion_base={'大脑半球': ['增高', '不清', '水肿带', '扩大', '变浅', '清楚', '尚清', '高密度影', '积血', '密度影', '增厚', '灶', '肿胀', '影', '变形', '欠清', '增宽', '变窄', '减低', '低密度影', '移位', '受压'],
             '基底节区': ['增高', '不清', '水肿带', '扩大', '变浅', '清楚', '尚清', '高密度影', '密度影', '灶', '肿胀', '影', '变形', '欠清', '增宽', '变窄', '减低', '低密度影', '受压'],
             '侧脑室': ['增高', '不清', '右偏', '水肿带', '扩大', '清楚', '尚清', '高密度影', '积血', '密度影', '灶', '肿胀', '影', '变形', '欠清', '增宽', '变窄', '减低', '低密度影', '移位', '受压'],
             '放射冠': ['增高', '密度影', '欠清', '灶', '增宽', '水肿带', '变浅', '清楚', '影', '高密度影', '减低', '低密度影', '受压', '移位'],
             '半卵圆中心': ['不清', '密度影', '欠清', '灶', '增宽', '扩大', '清楚', '尚清', '影', '高密度影', '减低', '低密度影'],
             '脑沟': ['增高', '不清', '密度影', '增厚', '变形', '增宽', '扩大', '变浅', '变窄', '肿胀', '影', '高密度影', '减低', '低密度影', '受压'],
             '中线': ['增高', '右偏', '扩大', '变浅', '清楚', '左移', '高密度影', '密度影', '增厚', '灶', '肿胀', '影', '变形', '欠清', '增宽', '左偏', '移位', '低密度影', '减低', '变窄', '受压'],
             '脑室': ['增高', '增厚', '密度影', '变形', '欠清', '灶', '增宽', '扩大', '出血', '清楚', '变窄', '肿胀', '影', '高密度影', '减低', '低密度影', '受压', '移位'],
             '脑裂': ['增高', '密度影', '增厚', '增宽', '扩大', '肿胀', '影', '高密度影', '低密度影'],
             '脑实质': ['增高', '密度影', '增厚', '灶', '增宽', '扩大', '肿胀', '左移', '高密度影', '移位', '低密度影', '受压', '减低'],
             '大脑镰': ['增高', '密度影', '增厚', '灶', '增宽', '左偏', '水肿带', '影', '高密度影', '低密度影'],
             '颞': ['增高', '不清', '密度影', '欠清', '尚清', '影', '高密度影', '低密度影'],
             '颅板': ['增高', '不清', '密度影', '增厚', '增宽', '影', '高密度影', '低密度影'],
             '顶叶': ['增高', '不清', '密度影', '欠清', '灶', '水肿带', '扩大', '变浅', '出血', '变窄', '清楚', '尚清', '肿胀', '影', '高密度影', '减低', '低密度影', '受压'],
             '脑组织': ['密度影', '变形', '肿胀', '高密度影', '减低', '低密度影', '受压'],
             '丘脑': ['密度影', '变形', '欠清', '水肿带', '出血', '清楚', '尚清', '影', '高密度影', '减低', '低密度影', '受压'],
             '小脑': ['增高', '不清', '密度影', '增厚', '欠清', '灶', '增宽', '水肿带', '清楚', '尚清', '肿胀', '影', '高密度影', '减低', '低密度影', '移位'],
             '病变': ['密度影', '增宽', '清楚', '减低', '低密度影'],
             '边界': ['不清', '密度影', '欠清', '灶', '增宽', '水肿带', '扩大', '清楚', '变窄', '尚清', '影', '减低', '低密度影', '受压'],
             '脑干': ['增高', '水肿带', '扩大', '清楚', '高密度影', '积血', '密度影', '增厚', '灶', '出血', '肿胀', '影', '变形', '欠清', '增宽', '移位', '低密度影', '减低', '受压'],
             '枕叶': ['不清', '密度影', '变形', '欠清', '灶', '水肿带', '清楚', '尚清', '肿胀', '影', '高密度影', '减低', '低密度影', '受压'],
             '额叶': ['增高', '不清', '密度影', '欠清', '灶', '增宽', '水肿带', '扩大', '变浅', '清楚', '变窄', '尚清', '肿胀', '影', '高密度影', '减低', '低密度影', '受压'],
             '脑回': ['增高', '密度影', '增厚', '灶', '增宽', '扩大', '肿胀', '影', '高密度影', '低密度影'],
             '基底节': ['不清', '密度影', '欠清', '扩大', '影', '高密度影', '减低', '低密度影'],
             '脑白质': ['增宽', '扩大', '影', '高密度影', '减低', '低密度影'],
             '颞部': ['增高', '肿胀', '影', '高密度影', '低密度影'],
             '小脑幕': ['增高', '密度影', '增厚', '增宽', '影', '高密度影', '变窄', '低密度影', '受压'],
             '第三脑室': ['增高', '积血', '扩大', '高密度影', '变窄', '受压', '移位'],
             '大脑': ['增高', '灶', '增宽', '影', '高密度影', '减低', '低密度影', '受压'],
             '小脑半球': ['增高', '不清', '密度影', '欠清', '灶', '增宽', '清楚', '高密度影', '减低', '低密度影', '受压'],
             '上颌窦': ['增厚', '密度影', '肿胀', '影', '高密度影', '低密度影'],
             '颞叶': ['增高', '密度影', '欠清', '灶', '水肿带', '扩大', '清楚', '影', '高密度影', '变窄', '低密度影', '减低'],
             '侧脑室前角': ['密度影', '清楚', '变窄', '低密度影', '受压'],
             '蝶窦': ['增高', '增厚', '密度影', '增宽', '影', '高密度影', '低密度影'],
             '骨质': ['增高', '增厚', '增宽', '肿胀', '影', '高密度影', '移位', '受压'],
             '侧脑室后角': ['密度影', '增宽', '扩大', '清楚', '影', '高密度影', '变窄', '低密度影', '受压'],
             '筛窦': ['低密度影', '增厚', '增高', '高密度影'],
             '颞枕叶': ['低密度影', '密度影', '影', '高密度影'],
             '软组织': ['增高', '密度影', '灶', '扩大', '肿胀', '影', '高密度影'],
             '第四脑室': ['增高', '影', '高密度影'],
             '鼻窦': ['增厚', '高密度影', '积血']
             }
lesion_word=['高密度影', '低密度影', '减低', "受压", '水肿带', '增宽', '增厚', '增高', '密度影', '变窄', '左移', '移位',
                '肿胀']
lesion_dict={
    '高密度影':1,
    '低密度影':2,
    '减低':3,
    '受压':4,
    '水肿带':5,
    '增宽':6,
    '增厚':7,
    '增高':8,
    '密度影':9,
    '变窄':10,
    '左移':11,
    '移位':12,
    '肿胀':13,
}
# Refine tissue-lesion relationships based on the specified lesion terms
lesion_new_base={}
for lesion in lesion_base:
    lesion_new_base[lesion]=[]
    for disease in lesion_base[lesion]:
        if disease in lesion_word:
            lesion_new_base[lesion].append(disease)
lesion_base = lesion_new_base
tissue_lesion_number = 0
lesion_lesion_number = 0
lesion_number=len(lesion_word) #13
print("A total of {} lesion keywords are included.".format(lesion_number))
Lesion_A=np.eye(lesion_number,k=0)
for tissue in lesion_base:
    tissue_relation_base=lesion_base[tissue]
    tissue_lesion_number += len(tissue_relation_base)
    tissue_relation_base_index=[]
    for x in tissue_relation_base:
        tissue_relation_base_index.append(lesion_dict[x])
    tissue_relation_pair= list(itertools.permutations(tissue_relation_base_index, 2))
    # print(len(tissue_relation_pair))
    lesion_lesion_number += len(tissue_relation_pair)
    for relation in tissue_relation_pair:
        if(Lesion_A[int(relation[0]-1)][int(relation[1]-1)]<1):
            Lesion_A[int(relation[0]-1)][int(relation[1]-1)]+=0.015

with open(r"/devdata/Brain_CT_Datasets/CTRG-Brain/label/data.json",'r',encoding='utf8') as fp1:
    dataset = json.load(fp1)["images"]

for index, sample in enumerate(dataset):
    # if index >= 5:
    #     break
    sample_name = str(sample["id"])

    # Summarize the tissue graph, lesion graph, and tissue-lesion graph into a relationship matrix, excluding global nodes at this stage.
    Relation_A = torch.tensor(np.eye(tissue_number + lesion_number,k=0))
    # Fill the top-left corner with tissue relationships
    for i in range(tissue_number):
        for j in range(tissue_number):
            Relation_A[i][j] = Tissue_A[i][j]
    # Fill the bottom-right corner with lesion relationships
    for i in range(lesion_number):
        for j in range(lesion_number):
            Relation_A[tissue_number+i][tissue_number+j] = Lesion_A[i][j]
    # Add symmetric tissue-lesion relationships to the top-right and bottom-left corners
    if str(sample["split"]) == "train":  # If it is a training set, use the tissue-lesion graph, as it is extracted based on the corpus.
        print("train", sample_name)
        # You can use simple rules to determine the relationships between the 19 tissue and 13 lesion terms in the current report. 
        # A value of 1 indicates a relationship, and 0 indicates no relationship. This will build a tissue-lesion relationship matrix. 
        # For convenience, we will directly load the pre-extracted file here.
        tissue_lesion_graph = torch.load("/devdata/Brain_CT_Datasets/CTRG-Brain/graph/Tissue_Lesion_Alignment/{}.pt".format(sample_name))  # 19*13
        Relation_A[0:19, 19:32] = tissue_lesion_graph
        Relation_A[19:32, 0:19] = tissue_lesion_graph.T
    else:  # For test and validation set data, we cannot access reports and thus cannot extract these relationships. Therefore, we use statistically derived general tissue-lesion relationships here.
        print("test/val", sample_name)
        for i in range(tissue_number):
            the_tissue_name = tissue_words[i]
            related_lesion_names = lesion_base[the_tissue_name]
            for lesion_name in related_lesion_names:
                matrix_index = tissue_number + (lesion_dict[lesion_name] - 1)  # The index of this lesion in the adjacency matrix should be placed after all tissues.
                # Add symmetric tissue-lesion relationships.
                Relation_A[i][matrix_index] = 0.5
                Relation_A[matrix_index][i] = 0.5
    Relation_tensor = Relation_A

    # Create a new 33x33 zero tensor to include the global node.
    Relation_tensor_new = torch.zeros(tissue_number + lesion_number + 1, tissue_number + lesion_number + 1)
    # Copy the data from the original tensor into the new tensor starting from the 1st row and 1st column.
    Relation_tensor_new[1:, 1:] = Relation_tensor
    # Set the element values to 1 in the 0th row and 1st column of the new matrix to represent a strong connection between the global node and other nodes.
    Relation_tensor_new[0, :] = 1  # Set all elements in the 0th row to 1.
    Relation_tensor_new[:, 0] = 1  # Set all elements in the 1st column to 1.

    file_path = f"/home/bjutcv/data/Yanzhaoshi/MRG_LLMs/llama/MEPNet/data_processing/Pathological_Graph/{sample_name}.pt"

    torch.save(Relation_tensor_new, file_path)



