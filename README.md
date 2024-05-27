# Large_Language_Model
Implementing a LLM from scratch

Progress:
1. Implemented a simple tokeniser
2. Implemented a tokeniser with special characters like <unk> and <endoftext>
3. Implemented Byte-Pair encoding based tokeniser as the final tokeniser for our model.
4. Created a custom dataset from a text file with inputs as texts of length=context size and outputs as the next word.
5. We want to train our model as a next word prediction model.
6. Created a dataloader for loading this data, given the arguments context size, batch size and stride.
