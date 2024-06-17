import numpy as np
class Attentionv1():
    def __init__(self, inputs):
        self.input=inputs
    def attn_scores(self, query):
        res=[]
        for el in self.input:
            dot_val=np.dot(query, el)
            res.append(dot_val)
        return res