import torch
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Tokenizer
from service.equation import WordEmbeddingBase
from base.utils import getPooledTensor


class Gpt2AttentionEmbedding(WordEmbeddingBase):

    embedding_model = GPT2Model.from_pretrained("openai-community/gpt2")
    """
    이부분은 원하면 tieWeight로 축소가능할듯,
    """
    inference_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    def __init__(
            self,
            word = None,
            tensor = None,
        ):
        super().__init__()

        self._word = word
        self._embedding = tensor
        if word:
            assert self.initialize_word()
        else:
            assert self.initialize_tensor()
    


    def initialize_word(self):
        encoded_input = self.__class__.tokenizer(self.word , return_tensors='pt', add_special_tokens=False)
        # print("encoded_input", encoded_input)
        output_tensor = self.__class__.embedding_model(**encoded_input)

        self.embedding = getPooledTensor(output_tensor.last_hidden_state)
        # print(output_tensor.last_hidden_state, self.embedding.shape, self.word)
        return True


    def initialize_tensor(self):

        # +/- 된 텐서를, 트랜스포머 한번 넣어주고,그값을 저 dictioanary에 내적해서, 가장큰값을 찾는건데,

        output_tensor = self.__class__.embedding_model(
            inputs_embeds = self.embedding.unsqueeze(0)
        ).last_hidden_state.squeeze(0)
        # print("output_tensor.shape", output_tensor.shape)
        self.embedding = output_tensor


        # result = self.__class__.inference_model(
        #     inputs_embeds = self.embedding.unsqueeze(0)
        # )
        # print("result", result.logits.shape)
        return True
        ...
    ...

PHYSICAL_DIVISION = 10
class VocabDictionary:

    def __init__(self, name = "None"):
        pass

    def embedding(self, vocab_list, embeddingModel : WordEmbeddingBase):
        self.words = vocab_list
        self.tensors = None


        # for vocab in vocab_list:
        #     emb = embeddingModel(vocab)
        #     if self.tensors is None:
        #         self.tensors = emb.embedding
        #     else:
        #         self.tensors = torch.cat([self.tensors, emb.embedding], dim = 0)

        # cls = embeddingModel.tokenizer("", add_special_tokens=True)
        # print(cls, '안녕하세요')
        # cls = embeddingModel.tokenizer.encode("")
        # print(cls, '안녕하세요')
        self.tensors = torch.cat(list(map(lambda x :  embeddingModel(x).embedding , self.words)), dim = 0)
        self.tensors = torch.nn.LayerNorm(normalized_shape=self.tensors.shape)(self.tensors)

        # print(self.tensors.shape)
        # torch.save(self.tensors, "my_dictionary2.pt")
        pass
    
    def query(self, query_tensor : torch.Tensor):
        """
        input: [1, emb] 
        """
        assert self.tensors is not None, "no tensor"
        print(query_tensor)
        result_tensor = torch.nn.Softmax(dim=1)(torch.matmul(query_tensor, self.tensors.T))
        print(result_tensor)
        result = torch.argmax(result_tensor, dim=1).data

        print(result, self.words[result])
        idxex = torch.topk(result_tensor, k = 3).indices.tolist()
        print("idxex::", idxex)
        candidates = list(map(lambda x : self.words[x], idxex[0]))
        print("candidates::", candidates)
        ...
    
    ...

if __name__ == "__main__":
    japan = Gpt2AttentionEmbedding("japan")
    korea = Gpt2AttentionEmbedding("korea")
    seoul = Gpt2AttentionEmbedding("seoul")

    tokyo = Gpt2AttentionEmbedding("tokyo")
    # newWord = korea - japan + tokyo
    newWord = (korea - japan) + seoul
    from gensim.models import Word2Vec
    """
    embedding 차원을 뭉개던 아니면
    """
    
    baseball = Gpt2AttentionEmbedding("baseball")
    bat = Gpt2AttentionEmbedding("bat")
    glove = Gpt2AttentionEmbedding("glove")
    foot = Gpt2AttentionEmbedding("foot")

    # glove = Gpt2AttentionEmbedding("glove")
    # glove = Gpt2AttentionEmbedding("glove")
    # glove = Gpt2AttentionEmbedding("glove")

    query_word = (baseball - bat) + foot
    # vocab_list = Word2Vec.load("/Users/leoyu/sangbumlikeagod/embedding/embedding/models/word2vec/word2vec").wv.index_to_key
    vocab_list = ["seoul", "korea" ,"japan", 'soccer',  '도쿄', '서울', 'baseball', 'bat', 'glove', 'foot', 'football', 'tokyo']
    # vocab_list = ["seoul", "korea" ,"japan", 'soccer',  'tokyo', 'emperor', 'reze', 'baseball', 'bat', 'glove', 'foot', 'football']
    vocab_vec = VocabDictionary()
    vocab_vec.embedding(vocab_list, Gpt2AttentionEmbedding)

    vocab_vec.query(newWord.embedding)
    vocab_vec.query(query_word.embedding)


    # print(vocab_vec.tensors.shape)


