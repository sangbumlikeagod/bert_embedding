import torch
from torch import nn
from bert.BertAttentionModel import BertAttentionEmbedding




if __name__ == "__main__":
    # tokyo =BertAttentionEmbedding("tokyo")
    # korea =BertAttentionEmbedding("korea")
    # seoul =BertAttentionEmbedding("seoul")


    tokyo =BertAttentionEmbedding("도쿄")
    korea =BertAttentionEmbedding("대한민국")
    seoul =BertAttentionEmbedding("서울")

    # [MASK] 토큰에 해당할 개수가 몇개인지 어케암?
    newToken = korea - seoul + tokyo