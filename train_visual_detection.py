from load_data import CelebaDetectionDataset
from model.detr import DETR
from torch.utils.data import DataLoader
import tqdm
import torch
from scipy.optimize import linear_sum_assignment
import json
import random
import cv2
import numpy as np

train_dataset = CelebaDetectionDataset()
valid_dataset = CelebaDetectionDataset(mode="valid")


def batch_linear_sum_assignment(pred_probs, preds, targets):
    """

    Parameters
    ----------
    pred_probs : 
        prediction probability of layout of certain query in dimension B x Q x 1
    preds : TYPE
        prediction of bbox in dimension B x Q x 4
    targets : TYPE
        prediction target layouts B x T x 4
    Returns
    -------
    row_idx : 
        in dimension (batch_idx, row_idx)
    col_idx : TYPE
        in dimension (batch_idx, col_idx)

    """
    
    batch_size = pred_probs.shape[0]
    batch_idx = []
    row_idices = []
    col_idices = []
    
    with torch.no_grad():
        for i in range(batch_size):
            try:
                target = targets[i,...] # T x 4
                num_t = target.shape[0]
            
                pred = preds[i] # Q x 4
                pred_prob = pred_probs[i] # Q x 1
                
                prob_cost_matrix = torch.cdist(pred_prob.sigmoid().detach(), torch.ones((num_t, 1), dtype=torch.float32, device=pred_prob.device))
                reg_cost_matrix = torch.cdist(pred.detach(), target) # Q x T
            
                cost_matrix = 0.5*prob_cost_matrix + 0.5*reg_cost_matrix
                ## Perform Hungarian Algorihtm
                row_idx, col_idx = linear_sum_assignment(cost_matrix.detach().cpu().numpy()) # T, T
                batch_idx += [i]*num_t 
                row_idices += row_idx.tolist()
                col_idices += col_idx.tolist()    
                
            except:
                pass
    
    batch_idx = torch.tensor(batch_idx).long()
    row_idices = torch.tensor(row_idices).long()
    col_idices = torch.tensor(col_idices).long()
    
    return (batch_idx, row_idices), (batch_idx, col_idices) 


class DiceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        intersection = (2 * pred * target).sum() 
        union = (pred + target).sum() + 1e-10
        return 1 - intersection / union


class IoULoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_bbox, target_bbox):
        ## pred_bbox N x 4
        ## target_bbox N x 4
        
        p_xmin, p_xmax  = pred_bbox[...,0], pred_bbox[...,0] + pred_bbox[...,2]
        p_ymin, p_ymax =  pred_bbox[...,1], pred_bbox[...,1] + pred_bbox[...,3]
        
        t_xmin, t_xmax  = target_bbox[...,0], target_bbox[...,0] + target_bbox[...,2]
        t_ymin, t_ymax =  target_bbox[...,1], target_bbox[...,1] + target_bbox[...,3]
        
        intersection_w = (torch.min(p_xmax, t_xmax) - torch.max(p_xmin, t_xmin)).relu()
        intersection_h = (torch.min(p_ymax, t_ymax) - torch.max(p_ymin, t_ymin)).relu()
        
        intersection = intersection_w * intersection_h
        union = (p_xmax - p_xmin)*(p_ymax - p_ymin) + (t_xmax - t_xmin)*(t_ymax - t_ymin) - intersection
        
        ious = (1 - intersection / union).mean()
        
        return ious


def compute_loss(pred_probs:torch.Tensor, pred_bbox:torch.Tensor, 
                 target_bbox: torch.Tensor, w = [1,1,1,1]) -> torch.Tensor:
    """
    
    Parameters
    ----------
    pred_prob : torch.Tensor
        torch.Tensor in dimension B x Q x 1 where Q = number of query
    pred_layout : torch.Tensor
        torch.Tensor in dimension B x Q x 4 where Q = number of query
    target_layout : torch.Tensor
        torch.Tensor in dimension B x T x 4 where T = number of target layout
    target_mask: torch.Tensor
        A boolean map represent the padding of target
        torch.Tensor in dimension B x T  where T = number of target layout

    Returns
    -------
    dict: Dictionary of Loss

    """
    
    mseloss = torch.nn.MSELoss()
    l1loss = torch.nn.L1Loss()
    diceloss = DiceLoss()
    bceloss = torch.nn.BCEWithLogitsLoss()
    iouloss = IoULoss()
    
    batch_size = pred_probs.shape[0]
    num_q = pred_probs.shape[1]

    
    pred_idx, target_idx = batch_linear_sum_assignment(pred_probs, pred_bbox, target_bbox)
    pred_bbox = pred_bbox[pred_idx]

    
    target_probs = torch.zeros((batch_size, num_q, 1), dtype=torch.float, device=pred_probs.device)
    target_probs[pred_idx] = 1.0
    
    
    pos_class_pred = pred_probs[pred_idx].view(-1)
    pos_target_probs = torch.ones_like(pos_class_pred)
    
    
    
    target_bbox = target_bbox[target_idx]

                      
    class_loss = bceloss(pred_probs, target_probs) + \
                 bceloss(pos_class_pred, pos_target_probs)
                 
                 
    dice_loss = diceloss(pred_probs.sigmoid(), target_probs)
    
    reg_loss = l1loss(pred_bbox, target_bbox) + torch.sqrt(mseloss(pred_bbox, target_bbox))
    
    iou_loss = iouloss(pred_bbox, target_bbox)
        
    loss_dict = {"Class": class_loss.item(), 
                 "Dice": dice_loss.item(), 
                 "BBox": reg_loss.item(), 
                 "IoU": iou_loss.item()
                 }
    
    total_loss = w[0]*class_loss + w[1]*dice_loss + w[2]*reg_loss  + w[3]*iou_loss

    
    return total_loss, loss_dict








def train(epochs=40):
    
    model = DETR(device="cuda:1")
    model.load_state_dict(torch.load("./detection_checkpoints/epoch_0_model.ckpt"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99997)
    history = []
    for i in range(epochs):
        dataloader = DataLoader(train_dataset, batch_size=24, shuffle=True)
        pbar = tqdm.tqdm(dataloader, total=len(dataloader))
        train_loss = 0
        n = 0
        for batch in pbar:
            optimizer.zero_grad()
            
            img, bbox = batch
            img = img.to(model.device)
            bbox = bbox.to(model.device)
            pred_bbox, pred_logits = model(img)

            loss, loss_dict = compute_loss(pred_logits, pred_bbox, bbox)
            # print(loss)
            train_loss += (loss.item() - train_loss) / (n+1)
            loss.backward()
            optimizer.step()
            loss_dict["Epoch Loss"] = train_loss
            loss_dict["Epochs"] = i
            history.append(loss_dict)
            pbar.set_postfix(loss_dict)
            
            n += 1
        
        if i % 5 == 0:
            torch.save(model.state_dict(), f"./detection_checkpoints/epoch_{i}_model.ckpt")
            with open("Detection_loss_hist.json", "w") as f:
                json.dump(history, f)
    
    torch.save(model.state_dict(), f"./detection_checkpoints/last_model.ckpt")



def evaluate():
    
    model = DETR(device="cuda:1")
    model.load_state_dict(torch.load("./detection_checkpoints/epoch_25_model.ckpt"))
    model.eval()
    

    dataloader = DataLoader(valid_dataset, batch_size=24, shuffle=True)
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    train_loss = 0
    train_loss_dict = {"Class": 0,
                 "Dice": 0,
                 "BBox": 0,
                 "IoU": 0
                 }
    n = 0
    for batch in pbar:

        img, bbox = batch
        img = img.to(model.device)
        bbox = bbox.to(model.device)
        B = img.shape[0]
        with torch.no_grad():
            pred_bbox, pred_logits = model(img.to(model.device))
        
        # pred_logits B x L
        # pred_bbox B x L x 4
        
        # if random.random() < 0.01:
        #     for b in range(B):
        #         idx = pred_logits[b].argmax()
        #         real_bbox = bbox[b][0] # 4
        #         chosen_bbox = pred_bbox[b][idx] # 4
        #         x, y, w, h = chosen_bbox.cpu().numpy()*256
        #         rx, ry, rw, rh = real_bbox.cpu().numpy()*256
        #         np_img = (img[b].permute(1,2,0)*255).cpu().contiguous().numpy()
        #         np_img = np.round(np_img).astype(np.uint8)
        #         np_img = cv2.rectangle(np_img, (round(rx),round(ry)),(round(rx+rw), round(ry+rh)), (0,0,255), 3)
        #         np_img = cv2.rectangle(np_img, (round(x),round(y)),(round(x+w), round(y+h)), (255,0,0), 3)
        #         name = int(1e10*random.random())
        #         cv2.imwrite(f"./detection_img/{name}.png", np_img[:,:,::-1])
            

        loss, loss_dict = compute_loss(pred_logits, pred_bbox, bbox)
        # print(loss)
        train_loss += (loss.item() - train_loss) / (n+1)
        
        for key, value in loss_dict.items():
            train_loss_dict[key] += (value - train_loss_dict[key]) / (n+1)
        train_loss_dict["Epoch Loss"] = train_loss
        pbar.set_postfix(train_loss_dict)
        
        n += 1
    
   


if __name__ == "__main__":
    
    evaluate()
