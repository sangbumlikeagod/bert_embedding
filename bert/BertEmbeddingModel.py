from transformers import BertModel, BertForMaskedLM
from transformers import AutoModelForMaskedLM

from transformers import BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings
import torch
from torch import nn

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
# text = "[CLS] korea's capital is seoul japan's capital is [MASK] [SEP]"
# encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False)
# print(encoded_input.input_ids[0].tolist())
# input_list = encoded_input.input_ids[0].tolist()



# 1 * 768
# print(output.pooler_output.shape)
# print(output.last_hidden_state.shape)


def getPooledTensor(tensor : torch.Tensor, mask_index : int = 0):
    """
    tensor, [batch, seq_len, hidden_size] 이 주어질떄를 가정

    return: [batch = 1, 1, hidden_size]
    """

    """
    CLS 토큰을 빼고 값 찾기
    """
    mask = torch.ones(tensor.shape)
    """
    사자
    """
    mask[:, mask_index, :] = 0
    """
    마스크된 텐서를 제작
    """
    masked_tensor = tensor * mask
    
    """
    [batch, vocab]
    """
    pooled_tensor = masked_tensor.sum(dim = 1) / mask.sum(dim = 1)
    return pooled_tensor



# output = model(**encoded_input)
# print(output)
# mask_index = encoded_input.input_ids[0].tolist().index(tokenizer.encode('[MASK]', add_special_tokens=False)[0])

# print(getPooledTensor(output.last_hidden_state).shape, end="끄어어어어어어어어어어어어어얶\n")
# from transformers import pipeline
# unmasker = pipeline('fill-mask', model='bert-base-uncased')
# unmasker("Hello I'm a [MASK] model.")

#BertForMaskedLM
# automodel = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
# output = automodel(**encoded_input)

# # answers = output.logits[:, mask_index].squeeze(0)
# nn.functional.softmax(answers, dim=0)
# candidates = torch.topk(answers, k = 10).indices.tolist()
# candidates = list(map(lambda x : tokenizer.decode(x), candidates))
# print(candidates)

def find_seq_len(word):
    lst = tokenizer.encode(word, add_special_tokens=False)
    print(f"{word}tokenize::{lst}")
    return lst


# print(tokenizer.decode(find_seq_len("한국")))
# find_seq_len("한국")
# find_seq_len("서울")
# find_seq_len("도쿄")
# find_seq_len("일본")



if __name__ == "__main__":
    embedding = model.embeddings

    #하튼 이거임
    tokens = tokenizer.encode('[MASK]', add_special_tokens=False)
    print(type(tokens), tokens)
    tsr = torch.tensor(tokens).unsqueeze(0)
    print(tsr, tsr.shape)
    out = embedding(torch.tensor(tokens).unsqueeze(0))
    print(out.shape)

    

    encoded_input = tokenizer("" , return_tensors='pt', add_special_tokens=True)
    inputs_id = encoded_input.input_ids[0].tolist()

    # getPooledTensor
    output_tensor = model(**encoded_input).last_hidden_state
    print(output_tensor)
    _embedding = getPooledTensor(output_tensor)
    print(_embedding)