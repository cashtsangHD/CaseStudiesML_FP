from load_data import CelebaRecognitionDataset
from model.vit import VIT
from torch.utils.data import DataLoader
import tqdm
import torch
import json

train_dataset = CelebaRecognitionDataset()
valid_dataset = CelebaRecognitionDataset(mode="valid")

def compute_clip_loss(embed1, embed2, temperature):
    
    B = embed1.shape[0]
    loss_fn = torch.nn.CrossEntropyLoss()
    
    embed1 = embed1 / embed1.norm(dim=-1, keepdim=True)
    embed2 = embed2 / embed2.norm(dim=-1, keepdim=True)
    
    M1 = torch.matmul(embed1, embed2.t()) * torch.exp(temperature)
    M2 = M1.t()
    
    label = torch.arange(0, B).to(embed1.device)
    loss = loss_fn(M1, label) + loss_fn(M2, label)
    
    return loss / 2


def compute_siglip_loss(embed1, embed2, temperature, bias):
    
    B = embed1.shape[0]
    loss_fn = torch.nn.LogSigmoid()
    
    embed1 = embed1 / (embed1.norm(dim=-1, keepdim=True) + 1e-6)
    embed2 = embed2 / (embed2.norm(dim=-1, keepdim=True) + 1e-6)
    
    M = torch.matmul(embed1, embed2.t()) * torch.exp(temperature) + bias

    label = 2*torch.eye(B, device=embed1.device) - torch.ones((B,B), device=embed1.device) 
    loss = -loss_fn(label*M)
    
    return loss.sum() / B
    

def eval_acc(embed1, embed2, temperature, bias=None):
    
    if bias is None:
        bias = 0
    
    B = embed1.shape[0]
    
    embed1 = embed1 / embed1.norm(dim=-1, keepdim=True)
    embed2 = embed2 / embed2.norm(dim=-1, keepdim=True)
    
    M1 = torch.matmul(embed1, embed2.t()) * torch.exp(temperature) + bias # B x B
    
    label = torch.arange(0, B).to(embed1.device)
    pred = torch.argmax(M1, dim=1)
    
    acc = (pred == label).sum()*100 / B
    
    return acc


def train(epochs=800):
    
    
    model = VIT()
    model.load_state_dict(torch.load("./checkpoints/epoch_370_model.ckpt"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99997)
    history_loss = []
    
    for i in range(epochs):
        dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        pbar = tqdm.tqdm(dataloader, total=len(dataloader))
        train_loss = 0
        n = 0
        for batch in pbar:
            optimizer.zero_grad()
            
            img1, img2 = batch
            img1 = img1.to(model.device)
            img2 = img2.to(model.device)
            img1_embed = model(img1)
            img2_embed = model(img2)
            
            # loss = compute_clip_loss(img1_embed, img2_embed, model.temperature)
            loss = compute_siglip_loss(img1_embed, img2_embed, model.temperature, model.bias)
            # print(loss)
            train_loss += (loss.item() - train_loss) / (n+1)
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"Loss": loss.item(), "Epoch Loss": train_loss})
            
            n += 1
        history_loss.append(train_loss)
        if i % 10 == 0:
            torch.save(model.state_dict(), f"./checkpoints/epoch_{i}_model.ckpt")
            with open("./train_hist.json", "w") as f:
                json.dump(history_loss, f)
    
    torch.save(model.state_dict(), f"./checkpoints/last_model.ckpt")


def evaluate():
    model = VIT(bias=False)
    model.load_state_dict(torch.load("./checkpoints/11-20-clip/last_model.ckpt"), )
    model.eval()
    
    dataloader = DataLoader(valid_dataset, batch_size=24, shuffle=True)
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    
    total_acc = 0
    n = 0
    with torch.no_grad():
        for batch in pbar:
            img1, img2 = batch
            img1 = img1.to(model.device)
            img2 = img2.to(model.device)
            img1_embed = model(img1)
            img2_embed = model(img2)
            
            acc = eval_acc(img1_embed, img2_embed, model.temperature, model.bias)
            total_acc += (acc.item() - total_acc) / (n+1)
            
            pbar.set_postfix({"Acc": acc.item(), "Mean Acc": total_acc})
            
            n += 1

if __name__ == "__main__":
    
    evaluate()
