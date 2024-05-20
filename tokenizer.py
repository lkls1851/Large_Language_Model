## We will use Python's regular expression library 're'
import numpy as np
import pandas as pd
import re

f=open('verdict.txt', 'r')
text=f.read()
num_char=len(text)
print(num_char,'\n')
print(text[:99], '\n')

## Splitting by punctuations
result=re.split(r'([:,;.!?"()\']|--|\s)', text)
print(result[:10])

## Removing whitespaces
result=[el.strip() for el in result if el.strip()]
