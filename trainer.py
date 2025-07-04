from t5 import T5,load_pretrained_weights
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from data_handler import DataIterator,tokenizer,collate_fn
import json 
import torch 
from torch.utils.data import DataLoader

from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

## embed dimension
d_dim = 512 
vocab_size = 32128
droput = 0.1
# use the shared embedding weights for both encoder and decoder 
tie_token_emb = True

enc_depth = 8
enc_heads = 6
enc_dim_head = 64
enc_mlp_mult = 2

dec_depth = 8
dec_heads = 6
dec_dim_head = 64
dec_mlp_mul = 2

num_epochs = 1
max_grad_norm = 1.0
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.eos_token_id or 1  # default

def train_epoch(model, dataloader):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["input_attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        _, loss = model(input_ids, labels, src_mask=attention_mask)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader):
    model.eval()
    total_loss = 0
    preds = []
    targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, labels, src_mask=attention_mask)
            total_loss += loss.item()


    return total_loss / len(dataloader)


if __name__ == "__main__":

    t5model = T5(dim=d_dim,
                enc_num_tokens=vocab_size,
                enc_depth=enc_depth,
                enc_heads=enc_heads,
                enc_dim_head=enc_dim_head,
                enc_mlp_mult=enc_mlp_mult,
                dec_num_tokens=vocab_size,
                dec_depth=dec_depth,
                dec_heads=dec_heads,
                dec_dim_head=dec_dim_head,
                dec_mlp_mult=dec_mlp_mul,
                dropout=droput,
                tie_token_emb=tie_token_emb)

    t5model_weights = t5model.state_dict()

    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    pretrained_model_weights = pretrained_model.state_dict()

    t5model_weights = load_pretrained_weights(t5model_weights,pretrained_model_weights)

    t5model.load_state_dict(t5model_weights)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t5model.to(device)

    with open("command.json","r") as f:
        data = json.load(f)

    train_size = int(len(data)*0.8)
    train_data = data[:train_size]
    valid_data = data[train_size:]

    train_dataset = DataIterator(train_data)
    valid_dataset = DataIterator(valid_data)

    train_loader = DataLoader(train_dataset,batch_size=8,shuffle=True,collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset,batch_size=8,shuffle=True,collate_fn=collate_fn)

    optimizer = Adam(t5model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        train_loss = train_epoch(t5model, train_loader)
        val_loss, bleu = validate(t5model, valid_loader)

        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")




