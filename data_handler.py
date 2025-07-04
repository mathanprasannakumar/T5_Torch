import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

class DataIterator(Dataset):

    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.data)}")

        item = self.data[idx]
        input_text = item['input']
        output_text = item['output']

        input_ids = tokenizer.encode(input_text)
        label_ids = tokenizer.encode(output_text)

        return { 'input_ids' :input_ids,'labels':label_ids}


def collate_fn(batch):

   input_ids =  [torch.tensor(item['input_ids']) for item in batch]
   labels = [torch.tensor(item['labels']) for item in batch]

   input_padded = torch.nn.utils.rnn.pad_sequence(input_ids,batch_first=True,padding_value=tokenizer.pad_token_id)

   input_attention_mask = input_padded.ne(tokenizer.pad_token_id).long()

   label_padded = torch.nn.utils.rnn.pad_sequence(labels,batch_first=True,padding_value=tokenizer.pad_token_id)

   label_padded[label_padded == tokenizer.pad_token_id] = -100

   return {'input_ids':input_padded,'input_attention_mask':input_attention_mask,
           'labels':label_padded}
