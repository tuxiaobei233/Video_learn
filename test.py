import os
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AutoModel, AutoTokenizer, AutoConfig
import torch
import json

# videos = json.loads(open("videos.json", "r").read())
tokenizer = BertTokenizer.from_pretrained(
    "nghuyong/ernie-3.0-base-zh", use_fast=True)


model = AutoModel.from_pretrained("nghuyong/ernie-3.0-base-zh")
"""
pad_embedding = model.embeddings.word_embeddings(torch.tensor([[0]]))
torch.save(pad_embedding, 'datasets/pad_embedding.pt')
cls_embedding = model.embeddings.word_embeddings(torch.tensor([[1]]))
torch.save(cls_embedding, 'datasets/cls_embedding.pt')
sep_embedding = model.embeddings.word_embeddings(torch.tensor([[2]]))
torch.save(sep_embedding, 'datasets/sep_embedding.pt')
"""


def get_video_feature(video_id, video_data_path):
    f = open(video_data_path +
             video_id + ".csv", "r").readlines()[1:]
    feature = []
    for line in f:
        feature.append([float(x) for x in line.split(",")])
    pt = torch.tensor(feature)
    pt = pt.unsqueeze(0)
    m = torch.nn.AdaptiveAvgPool2d((25, 768))
    pt = m(pt)
    return pt


def get_text_feature(video_id, text_data_path):
    text = open(text_data_path + video_id + ".txt", "r").read()
    text_inputs = tokenizer(text, padding="max_length",
                            max_length=485, truncation=True, add_special_tokens=False, return_tensors="pt")
    print(text_inputs)
    word_embeddings = model.embeddings.word_embeddings(
        text_inputs["input_ids"])
    return word_embeddings, text_inputs


def get_text_feature1(video_id, text_data_path):

    text = open(text_data_path + video_id + ".txt", "r").read()
    if len(text) == 0:
        return torch.zeros((1, 1, 768))
    text_inputs = tokenizer(
        text, add_special_tokens=False, return_tensors="pt")
    word_embeddings = model.embeddings.word_embeddings(
        text_inputs["input_ids"])
    return word_embeddings


def get_text_feature2(video_id, text_data_path="datasets/text_feature/"):
    word_embeddings = torch.load(text_data_path + video_id + ".pt")
    if word_embeddings.shape[1] >= 485:
        return word_embeddings[:, :485, :], torch.tensor([[1] * 485])
    else:
        pad_embedding = torch.load("datasets/pad_embedding.pt")
        attention_mask = torch.tensor([[1] * word_embeddings.shape[1]])
        p = torch.tensor([[0] * (485 - word_embeddings.shape[1])])
        print(attention_mask)
        print(attention_mask.shape)
        print(p)
        print(p.shape)
        attention_mask = torch.cat((attention_mask, p), 1)
        pad_embedding = pad_embedding.repeat(
            word_embeddings.shape[0], 485 - word_embeddings.shape[1], 1)
        word_embeddings = torch.cat((word_embeddings, pad_embedding), 1)
    return word_embeddings, attention_mask


def calc_video(video_id, video_data_path="datasets/video_feature/", text_data_path="datasets/text/"):
    cls_embedding = torch.load("datasets/cls_embedding.pt")
    sep_embedding = torch.load("datasets/sep_embedding.pt")
    res_inputs = {}
    video_feature = get_video_feature(video_id, video_data_path)
    text_feature, text_inputs = get_text_feature(video_id, text_data_path)
    token_type_ids = torch.tensor([[0] * 27 + [1] * 485])
    res_inputs['token_type_ids'] = token_type_ids
    attention_mask = torch.cat((torch.tensor(
        [[1] * 27]), text_inputs['attention_mask']), 1)
    res_inputs['attention_mask'] = attention_mask
    res_inputs['inputs_embeds'] = torch.cat(
        (cls_embedding, video_feature, sep_embedding, text_feature), 1)
    print(video_feature.shape)
    print(text_feature.shape)
    print(res_inputs['token_type_ids'].shape)
    print(res_inputs['attention_mask'].shape)
    print(res_inputs['inputs_embeds'].shape)
    return res_inputs


def get_feature(video_ids, video_data_path="datasets/video_feature/", text_data_path="datasets/text/"):
    all_res_inputs = {
        'token_type_ids': torch.tensor([]),
        'attention_mask': torch.tensor([]),
        'inputs_embeds': torch.tensor([])
    }
    for video_id in video_ids:
        res_inputs = calc_video(video_id)
        for key in all_res_inputs:
            all_res_inputs[key] = torch.cat(
                (all_res_inputs[key], res_inputs[key]), 0)
    return all_res_inputs


video_id = "douyin_6559701594739313923"
text_data_path = "datasets/text/"
print(video_id)
a = get_text_feature1(video_id, text_data_path)
torch.save(a, 'datasets/text_feature/' + video_id + '.pt')
"""
text_data_path = "datasets/text/"
files = os.listdir(text_data_path)
for file in files:
    video_id = file[:-4]
    print(video_id)
    a = get_text_feature1(video_id, text_data_path)
    torch.save(a, 'datasets/text_feature/' + video_id + '.pt')
"""
"""
sequence = "四川大学网络空间安全学院"

model_inputs = tokenizer(sequence, padding="max_length",
                         max_length=485, truncation=True, add_special_tokens=False, return_tensors="pt")
print(model_inputs)

# tokens = sequences
# ids = tokenizer.convert_tokens_to_ids(tokens)
# print(ids)

# decoded_string = tokenizer.decode(ids)
# print(decoded_string)
# print(tokenizer.decode(model_inputs["input_ids"]))

word_embeddings = model.embeddings.word_embeddings(model_inputs["input_ids"])
print(word_embeddings)
outputs = model(**model_inputs)
pooler_output = outputs.pooler_output
print(pooler_output.shape)
"""
