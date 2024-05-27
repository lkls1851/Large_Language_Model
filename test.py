import os
import re
from tokeniser import SimpleTokeniser
from tokeniserv2 import SimpleTokeniserv2
from bpe_tokeniser import BPETokeniser

path_to_txt='verdict.txt'

tokeniser=SimpleTokeniser(path_to_text=path_to_txt)

tokeniser2=SimpleTokeniserv2(path_to_text=path_to_txt)

bpe_tokeniser=BPETokeniser()

## What is the encoded ID for 'is'?
print('Test:1 \n')
print(tokeniser.encode('is'), '\n')


## What is the word corresponding to the token 111

print('Test:2 \n')

print(tokeniser.decode(111), '\n')


## Encode a text

txt='It is the last painting'

print('Test:3 \n')

print(tokeniser.encode(txt), '\n')


## Convert a list of IDs to text

ids=[111, 876, 992, 0, 4]
print('Test:4 \n')

print(tokeniser.decode(ids), '\n')



## Tests for tokeniser v2

text='Hi how are u?'
text2='I am fine. Thanks for asking.'

txt=' <endoftext> '.join((text, text2))

print('Test:5 \n')

print(tokeniser2.encode(txt), '\n')


## Decode a list:
ids=[1139, 571, 182, 1139, 10, 1140, 53, 1139, 1139, 7, 1139, 468, 1139, 7]

print('Test:6 \n')

print(tokeniser2.decode(ids), '\n')


id=1139

print('Test:7 \n')

print(tokeniser2.decode(id), '\n')



## Encode and Decode using bpe_tokeniser
## Encode a text

txt='Hare Krishna Om Namaha Shivaaya'

print('Test:8 \n')

print(bpe_tokeniser.encode(txt), '\n')
40904090
print('Test:9 \n')

ids=[39, 533, 38594, 16543, 17871, 12236, 33995, 11729]
print(bpe_tokeniser.decode(ids), '\n')


print('Test:10 \n')

print(bpe_tokeniser.encode('cvSAGD WFUNWB XJVCBSJDVBksjahd'), '\n')

print('Test:11 \n')

print(bpe_tokeniser.decode([33967, 4090, 45113, 370, 42296, 45607, 1395, 41, 15922, 4462, 37882, 44526, 591, 31558, 67] ), '\n')


print('Test:12 \n')
print(bpe_tokeniser.decode(33967), '\n')


print('Test:13 \n')

from custom_dataset import MyDataset

mydataloader=MyDataset(path_to_text=path_to_txt, context_size=4)

count=0
for i, (x,y) in enumerate(mydataloader):
    count+=1
    print(x, '---->', y)
    if count==5:
        break
    else:
        continue


print('Test:14 \n')

from gpt_dataset import GPTDataset
my_dataset=GPTDataset(path_to_text=path_to_txt, context_size=5, stride=1)

for i, (x,y) in enumerate(my_dataset):
    if i<5:
        print(x, '--->', y)
    else:
        break


print('\n')
print('Test:15 \n')

from gpt_dataset import GPTDataset
my_dataset=GPTDataset(path_to_text=path_to_txt, context_size=5, stride=2)

for i, (x,y) in enumerate(my_dataset):
    if i<5:
        print(x, '--->', y)
    else:
        break


print('\n')
print('Test:16 \n')
## Test Dataloader

import torch
from torch.utils.data import DataLoader

dataloader=DataLoader(dataset=my_dataset, batch_size=1, shuffle=False)
count=0
for i, (x, y) in enumerate(dataloader):
    if count<=5:
        print(x, ' ---> ', y, '\n')
        count+=1
    else:
        break


print('\n')
print('Test:17 \n')

from embedding import Embedding

txt='I am susmit'
ids=bpe_tokeniser.encode(txt)
embed=Embedding(ids=ids, num_emb_space=5)
print(embed.weights())


print('\n', 'Test:18 \n')

## Extract the embedding for a specific id

print(embed.forward(2))