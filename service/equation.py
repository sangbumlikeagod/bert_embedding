import torch
from torch import nn
from gensim.models import Word2Vec


class WordEmbeddingBase:

    @property
    def word(self):
        return self._word
    
    @word.setter
    def word(self, data : str):
        self._word = data

    @property
    def embedding(self):
        return self._embedding
    
    @embedding.setter
    def embedding(self, data : torch.Tensor):
        self._embedding = data
    def __init__(self):
        ...

    def __add__(self, other):
        newTensor = torch.add(self.embedding, other.embedding)

        return self.__class__(
            tensor=newTensor
        )


    def __sub__(self, other):
        newTensor = torch.add(self.embedding, torch.neg(other.embedding))
        return self.__class__(
            tensor=newTensor
        )
    



class WordEmbedding(WordEmbeddingBase):
    model = Word2Vec.load("/Users/leoyu/sangbumlikeagod/embedding/embedding/models/word2vec/word2vec")
    model_words = {}
    model_tensors = None


        

    def __init__(
            self,
            word = None,
            tensor = None,
        ):
        if not self.__class__.model_words:
            for idx, val in enumerate(self.__class__.model.wv.index_to_key):
                self.__class__.model_words[val] = idx
            ...
        if self.__class__.model_tensors is None:
            self.__class__.model_tensors = torch.tensor(self.__class__.model.wv.vectors)

        pass
        super().__init__()
        self._word = word
        self._embedding = tensor


        if word:
            assert self.initialize_word()
        else:
            assert self.initialize_tensor()
    

    def initialize_tensor(self):
        if self.word:
            return True
        i_tensor = nn.functional.softmax(
            torch.matmul(
            self.embedding,
            self.__class__.model_tensors.T
            ), 
        dim=0)
        
        idxex = torch.topk(i_tensor, k = 10).indices.tolist()
        self.candidates = list(map(lambda x : self.__class__.model.wv.index_to_key[x], idxex))
        idx = torch.argmax(i_tensor, dim=0).data
        self.word = self.__class__.model.wv.index_to_key[idx]
        return True
    
    def initialize_word(self):
        word_idx = self.__class__.model_words.get(self.word, -1)
        if word_idx == -1:
            return False
        self.embedding = self.__class__.model_tensors[word_idx]

        return True
    

    # def __add__(self, other):
    #     newTensor = torch.add(self.embedding, other.embedding)

    #     return WordEmbedding(
    #         tensor=newTensor
    #     )


    # def __sub__(self, other):
    #     newTensor = torch.add(self.embedding, torch.neg(other.embedding))
    #     return WordEmbedding(
    #         tensor=newTensor
    #     )
    

if __name__ == "__main__":

    seoul = WordEmbedding(word="서울")
    korea = WordEmbedding(word="대한민국")
    japan = WordEmbedding(word="일본")
    tokyo = WordEmbedding(word="도쿄")
    china = WordEmbedding(word="중국")


    new_token = korea - seoul + japan
    print(new_token.word, new_token.candidates)


    new_token_2 = korea + tokyo  - seoul
    # new_token_2 = korea - seoul + tokyo
    print(new_token_2.word, new_token_2.candidates)

    park = WordEmbedding(word="박찬호")
    baseball = WordEmbedding(word="야구")
    soccer = WordEmbedding(word="축구")
    new_token_3 = park - baseball + soccer
    print(new_token_3.candidates)

    ironman = WordEmbedding(word="아이언맨")
    marble = WordEmbedding(word="마블")
    dc = WordEmbedding(word="DC")
    new_token_4 = ironman - marble + dc
    print(new_token_4.candidates)

    ...
