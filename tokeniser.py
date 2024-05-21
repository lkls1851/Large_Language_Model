import numpy as np
import re

class SimpleTokeniser():
    def __init__(self, path_to_text):
        self.path=path_to_text
        with open(self.path, 'r') as f:
            self.vocab=f.read()
        self.red_tokens=re.split(r'([:,;.!?"()\']|--|\s)', self.vocab)
        self.tokens=sorted(list(set([el.strip() for el in self.red_tokens if el.strip()])))
        self.hashmap={token:index for index,token in enumerate(self.tokens)}
        self.inverse_hash={index:token for index, token in enumerate(self.tokens)}

    def encode(self, x):
        if x not in self.hashmap.keys():
            return -1
        
        else:
            return self.hashmap[x]
    
    def decode(self, id):
        if id not in self.inverse_hash.keys():
            return -1
        else:
            return self.inverse_hash[id]


