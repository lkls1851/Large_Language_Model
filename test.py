import os
import re
from tokeniser import SimpleTokeniser

path_to_txt='verdict.txt'

tokeniser=SimpleTokeniser(path_to_text=path_to_txt)


## What is the encoded ID for 'is'?

print(tokeniser.encode('is'), '\n')


## What is the word corresponding to the token 111


print(tokeniser.decode(111), '\n')


## Encode a text

txt='It is the last painting'

print(tokeniser.encode(txt), '\n')


## Convert a list of IDs to text

ids=[111, 876, 992, 0, 4]

print(tokeniser.decode(ids))
