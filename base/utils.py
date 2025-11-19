import torch
from torch import nn


def getPooledTensor(tensor : torch.Tensor, mask_index : int = -1):
    """
    tensor, [batch = 1, seq_len, hidden_size] 이 주어질떄를 가정

    return: [batch = 1, 1, hidden_size]
    """

    """

    []
    """

    mask = torch.ones(tensor.shape)
    """
    사자
    """
    if mask_index != -1:
        mask[:, mask_index, :] = 0
    # print(mask)
    """
    마스크된 텐서를 제작
    """
    masked_tensor = tensor * mask
    
    """
    [batch, vocab]
    """
    pooled_tensor = masked_tensor.sum(dim = 1) / mask.sum(dim = 1)
    return torch.nn.functional.layer_norm(pooled_tensor, pooled_tensor.shape)


if __name__ == "__main__":
    test_tensor = torch.Tensor([[[1, 2, 5], [3, 4, 6], [7, 8, 9]]])
    print(test_tensor.shape)
    pooled_tensor =     (test_tensor)
    print(pooled_tensor)