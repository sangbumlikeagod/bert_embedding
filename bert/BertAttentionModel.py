from service.equation import WordEmbeddingBase
from transformers import BertModel, BertForMaskedLM, BertTokenizer
from .BertEmbeddingModel import getPooledTensor
import torch
from torch import nn

class BertAttentionEmbedding(WordEmbeddingBase):
    embedding_model = BertModel.from_pretrained('bert-base-uncased')
    inference_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    mask_token = "[MASK]"


    def __init__(
        self,
        word : str = None,
        tensor : torch.Tensor = None,
        num_of_mask : int = 2,
    ):
        self._word = word
        self._embedding = tensor
        self.num_of_mask = num_of_mask
        
        super().__init__()
        if word:
            assert self.initialize_word()
        else:
            assert self.initialize_tensor()


    def initialize_tensor(self):
        # 검증필요

        embedding_model = self.__class__.embedding_model.embeddings
        self.cls = self.__class__.tokenizer.encode(  
                "[CLS]", add_special_tokens=False
        )
        self.tokens = self.__class__.tokenizer.encode(  
                self.__class__.mask_token, add_special_tokens=False
        )
        self.sep = self.__class__.tokenizer.encode(  
                "[SEP]", add_special_tokens=False
        )
        
        cls_tensor = embedding_model(
            torch.tensor(self.cls).unsqueeze(0)
        )
        cls_tensor = cls_tensor.view([1, -1])

        mask_tensor = embedding_model(
            torch.tensor(self.tokens).unsqueeze(0)
        )
        mask_tensor = mask_tensor.view([1, -1])

        sep_tensor = embedding_model(
            torch.tensor(self.sep).unsqueeze(0)
        )
        sep_tensor = sep_tensor.view([1, -1])


        # 여기도 검증필요
        # inference_tensor = torch.cat([self.embedding.clone() for i in range(self.num_of_mask + 1) ], dim=0).unsqueeze(0)
        inference_tensor = torch.cat(
            [cls_tensor] + 
            [self.embedding] + 
            [mask_tensor.clone() for i in range(self.num_of_mask)] + 
            [sep_tensor], dim=0
        ).unsqueeze(0)

        output = self.__class__.inference_model(
            inputs_embeds=inference_tensor,
        )

        # answers = output.logits[:, 1].squeeze(0)
        # nn.functional.softmax(answers, dim=0)
        # # candidates = list(map(lambda x : , idxex))
        # candidates = torch.topk(answers, k = 10).indices.tolist()
        # print(candidates)
        # candidates = list(map(lambda x : self.__class__.tokenizer.decode(x), candidates))
        # print(candidates)

        answerss = []
        for i in range(2, self.num_of_mask + 2):
            ith_logits = output.logits[:, i].squeeze(0)
            print(ith_logits.shape)
            ith_logits = nn.functional.softmax(ith_logits, dim=0)
            candidates =torch.topk(ith_logits, k = 20).indices.tolist()
            candidates = list(map(lambda x : self.__class__.tokenizer.decode(x), candidates))
            print(candidates)
            maxis = torch.argmax(ith_logits, dim=0).data
            answerss.append(maxis)
            
        print(f"result of {self.num_of_mask} tokens", self.__class__.tokenizer.decode(answerss))
        
        return True

    # 어텐션까지 한 버전
    def initialize_word(self):
        encoded_input = self.__class__.tokenizer(self.word , return_tensors='pt', add_special_tokens=True)
        self.inputs_id = encoded_input.input_ids[0].tolist()

        # getPooledTensor
        output_tensor = self.__class__.embedding_model(**encoded_input).last_hidden_state
        self._embedding = getPooledTensor(output_tensor)
        print(self.embedding.shape)
        return True
        ...


    # 어텐션 안한버전
    # def initialize_word(self):
    #     encoded_input = self.__class__.tokenizer.encode(
    #         self.word, return_tensors='pt', add_special_tokens=True
    #     )
    #     # self.inputs_id = encoded_input.input_ids[0].tolist()
    #     # getPooledTensor
    #     embedding_model = self.__class__.embedding_model.embeddings
    #     mask_tensor = embedding_model(
    #         torch.tensor(encoded_input).unsqueeze(0)
    #     )
    #     #[1, seq_len, embed]
    #     mask_tensor = mask_tensor.squeeze(0)
    #     self._embedding = getPooledTensor(mask_tensor)

    #     print(self.embedding.shape)
    #     return True

        