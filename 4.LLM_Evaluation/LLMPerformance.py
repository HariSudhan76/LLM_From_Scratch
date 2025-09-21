#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import nbimporter
from GPT_Architecture import GPTModel
GPT_CONFIG_124M={
 "vocab_size":50257,
 "context_length":256,
 "emb_dim":768,
 "n_heads":12,
 "n_layers":12,
 "drop_rate":0.1,
 "qkv_bias":False
}


# In[3]:


from pypdf import PdfReader
reader = PdfReader("E:\LLM\LLM from Scratch\war-and-peace.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text()


# In[4]:


print(text[-99:])


# In[5]:


tot_char = len(text)
print(tot_char)


# In[6]:


###GPT Predict next token
def generate_text_simple(model,idx,max_new_tokens,context_size):
    #idx is (batch,n_tokens) array of indices in the current context
    # tensor([[8496,  612,  318],
    #     [8117,  318,  257]])

    for _ in range(max_new_tokens): ##Iteration repeated until reach max_new_tokens

        #Crop current context if it exceeds the supported context size
        #Eg. if LLM supports only 5 tokens, and the context size is 10 
        # then only the last 5 tokens are used as context
        idx_cond = idx[:,-context_size:]

        #Get the prediction
        with torch.no_grad():
            logits = model(idx_cond) ##Dimension are -> batch, n_tokens, vocab_size

        #Focus only on the last time step
        # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:,-1,:] ##-1 in n_tokens means it will pick only last n_tokens for that batch

        #Apply softmax to get probabilities
        probas = torch.softmax(logits, dim=-1) #batch, vocab_size
        ##Softmax is not necessary but to give the intuition and understand the highest value of word probab we use
        ##Softmax applied along every batch that is the rows

        #Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) #(batch,1)

        #Append sampled index to the running sequence
        idx = torch.cat((idx,idx_next), dim=1) #(batch, n_tokens+1)

    return idx



# In[7]:


import tiktoken

def text_to_token_ids(text,tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) #Add batch dim
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    flat = token_ids.squeeze(0) #Remove batch dimension
    return tokenizer.decode(flat.tolist())

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=GPTModel(GPT_CONFIG_124M),
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)


# In[8]:


def calc_loss_batch(input_batch,target_batch,model,device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten())
    return loss 

def calc_loss_loader(data_loader, model, device, num_batches=None): ##Calculate loss for all of the batches
    total_loss=0.
    if len(data_loader) ==0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)

    else:
        #Reduce the number of batches to match the total number of batches in the data loader
        #if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))

    for i,(input_batch, target_batch) in enumerate(data_loader):
        if i<num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss/num_batches



# ## Training Loop For the LLM

# In[9]:


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter,
                       start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    #Main training loop
    for epoch in range(num_epochs):
        model.train() #Set model to training mode

        for input_batch, target_batch in train_loader: #Train_loader split to input and target batch
            optimizer.zero_grad() #Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch,target_batch,model,device)
            loss.backward() #Calculate loss gradients, calculate backward pass for all the parameters
            optimizer.step() #Update model weights using loss gradients
            tokens_seen += input_batch.numel() #Returns the total number of elements (or tokens) in the input_batch
            global_step += 1

            #Optional evaluation step #Show only on specific frequency 
            ##Just prints every single epoch when it reaches specific frequency
            if global_step % eval_freq ==0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )
    return train_losses, val_losses, track_tokens_seen


# #### evaluate_model function calculates the loss over the training and validation set while ensuring the model is in evaluation mode with gradient tracking and dropout disabled when calculating the loss over the training and validation sets

# In[10]:


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
        val_loss = calc_loss_loader(val_loader,model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


# #### generate_and_print_sample function is a convenience function that we use to track whether the model imporves during the training.
# #### In particular, the generate_and_print_sample func takes a text snippet (start_context) as input, converts it into token IDs, and feeds it to the LLM to generate a text sample using the generate_text_simple func we used earlier

# In[11]:


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model = model, idx = encoded, 
            max_new_tokens = 50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids,tokenizer)
    print(decoded_text.replace("\n"," ")) #Compact print format
    model.train()


# In[42]:


import torch
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
##2. Tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Example dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=32):
        self.examples = []
        for t in texts:
            ids = tokenizer.encode(t, allowed_special={'<|endoftext|>'})
            if len(ids) < 2:  # skip empty/too short
                continue

            # Truncate or pad
            for i in range(0, max(1, len(ids) - 1), block_size):
                input_chunk = ids[i:i+block_size]
                target_chunk = ids[i+1:i+block_size+1]

                # Pad if needed
                if len(input_chunk) < block_size:
                    input_chunk += [tokenizer.eot_token] * (block_size - len(input_chunk))
                if len(target_chunk) < block_size:
                    target_chunk += [tokenizer.eot_token] * (block_size - len(target_chunk))

                input_ids = torch.tensor(input_chunk)
                target_ids = torch.tensor(target_chunk)
                self.examples.append((input_ids, target_ids))


    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Some dummy training text (replace with your dataset)
texts = ["This is good example"]

# Build datasets & loaders
train_dataset = TextDataset(texts, tokenizer, block_size=16)
val_dataset   = TextDataset(texts, tokenizer, block_size=16)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=2)


# In[43]:


print("Train dataset size:", len(train_dataset))
print("Val dataset size:", len(val_dataset))


# Using AdamW optimizers

# In[44]:


import time
start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader,optimizer,device,
    num_epochs = num_epochs, eval_freq=5, eval_iter=5,
    start_context="War is always", tokenizer=tokenizer
)
end_time = time.time()
execution_time_min = (end_time - start_time)/60
print(f"Training completed in {execution_time_min:.2f} minutes.")


# ### Output of Argmax will be deterministic, to make it more creative and get the result in a probablistic manner we go for 2 techniques
# ## Techinique 1 -  Testing Temperature Scaling

# In[15]:


vocab = {
    'closer': 0,
    'every' : 1,
    'effort': 2,
    'forward': 3,
    'inches': 4,
    'moves': 5,
    'pizza': 6,
    'towards': 7,
    'you': 8,

}

inverse_vocab = {v:k for k, v in vocab.items()}


# In[16]:


next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)


# In[17]:


probas = torch.softmax(next_token_logits, dim=0)
print(probas)

next_token_id = torch.argmax(probas).item()
print(next_token_id)
print(inverse_vocab[next_token_id])


# Now will replace argmax function with multinomial function and see the result
# 
# Higher T(temperature) - More creativity high risk\
# Lower T(temperature) - More Deterministic similar to argmax

# In[18]:


torch.manual_seed(43)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])


# In[19]:


def print_sample(probas):
    torch.manual_seed(43) #Manual seed for reproducibility
    sample = [torch.multinomial(probas,num_samples=1).item() for i in range(1_000)]
    sample_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sample_ids):
        print(f"{freq} x {inverse_vocab[i]}")

print_sample(probas)


# ### High temp can lead to more diverse tokens but can lead to grammatically incorrect or non sensical tokens to overcome this we go for 2nd technique
# ## Technique 2 - Top K - Sampling

# In[20]:


next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
)


# In[21]:


top_k = 3 ##Number of tokens required to be on the top
top_logits, top_pos = torch.topk(next_token_logits, top_k)
print(f"Top Logits: {top_logits}")
print(f"Top positions: {top_pos}")


# In[22]:


new_logits = torch.where(
    condition=next_token_logits < top_logits[-1],
    input = torch.tensor(float("-inf")),
    other = next_token_logits
)
print(new_logits)


# In[23]:


topk_probas = torch.softmax(new_logits,dim=0)
print(topk_probas)


# ### Combine Temp scaling and Top k

# In[31]:


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    #For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens): ##Continue until max number of new token limit is reached
        idx_cond = idx[:,-context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]

        #New: Filter logits with top_k sampling
        if top_k is not None:
            #Keep only the opt_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:,-1]
            logits = torch.where(logits<min_val.unsqueeze(-1),
                                 torch.tensor(float("-inf"),device=logits.device),
                                              logits
                                )

        #New: Apply temperature scaling 
        if temperature >0.0:
            logits = logits / temperature

            #Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1) #(Batch size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(batch_size, 1)

        #Otherwise same as before: get idx of the vocab entry with the highest logits value, this applies when temperature value is not specified
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True) #(batch_size, 1)

        if eos_id is not None and (idx_next == eos_id).any(): #Stop generating early if end-of-sequence token is encountered and eos_id is specified then stop early
            logits[:, eos_id] = float("-inf")
            break

        #Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) #(batch_size, num_tokens+1)

    return idx




# In[41]:


torch.manual_seed(45)
eos_token_id = tokenizer.encode(".", disallowed_special=())[0]
token_ids = generate(
    model = model,
    idx = text_to_token_ids("Transformers are powerful", tokenizer),
    max_new_tokens= 15,
    context_size= GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=0.9,
    eos_id=eos_token_id
)

print(f"Output text: {token_ids_to_text(token_ids,tokenizer)}")


# ### Saving and Loading Model weights in Pytorch

# In[45]:


model = GPTModel(GPT_CONFIG_124M)
torch.save(model.state_dict(), "model.pth")


# In[47]:


model.load_state_dict(torch.load("model.pth"))
model.eval()


# #### Saving and loading Model optimizer weights

# In[48]:


optimizer = torch.optim.AdamW(model.parameters(),lr=0.0004,weight_decay=0.1)
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
},
           "model_and_optimizer.pth")


# In[50]:


checkpoint = torch.load("model_and_optimizer.pth")
model = GPTModel(GPT_CONFIG_124M)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
model.train()

# --- script entrypoint ---
if __name__ == "__main__":
    # This block ONLY runs if you execute:
    #   python LLMPerformance.py
    # But it will NOT run if imported.
    
    print("Starting training...")
    # train loop here
    # e.g. train_model_simple(...)