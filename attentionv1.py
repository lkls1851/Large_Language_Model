import numpy as np
import torch
class Attentionv1():
    def __init__(self, inputs):
        self.input=inputs
    def attn_scores(self, query):
        res=[]
        for el in self.input:
            dot_val=torch.dot(query, el)
            np_val=dot_val.detach().numpy()
            np_val=np.float32(np_val)
            res.append(np_val)
        res=torch.tensor(res)
        return res