import os
import re
from tokeniser import SimpleTokeniser
from tokeniserv2 import SimpleTokeniserv2

path_to_txt='verdict.txt'

tokeniser=SimpleTokeniser(path_to_text=path_to_txt)

tokeniser2=SimpleTokeniserv2(path_to_text=path_to_txt)


## What is the encoded ID for 'is'?

print(tokeniser.encode('is'), '\n')


## What is the word corresponding to the token 111


print(tokeniser.decode(111), '\n')


## Encode a text

txt='It is the last painting'

print(tokeniser.encode(txt), '\n')


## Convert a list of IDs to text

ids=[111, 876, 992, 0, 4]

print(tokeniser.decode(ids), '\n')



## Tests for tokeniser v2

text='Hi how are u?'
text2='I am fine. Thanks for asking.'

txt=' <endoftext> '.join((text, text2))

print(tokeniser2.encode(txt), '\n')


## Decode a list:
ids=[1139, 571, 182, 1139, 10, 1140, 53, 1139, 1139, 7, 1139, 468, 1139, 7]

print(tokeniser2.decode(ids), '\n')


id=1139

print(tokeniser2.decode(id), '\n')