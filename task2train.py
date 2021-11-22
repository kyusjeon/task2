import numpy as np
import torch
from task2utils import *
##Cross Validation
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

# main
def train(dataloader, model, loss, device, num_epochs, batch_size, max_lr, num_workers, n_folds = 5):
    results = {}
    folds = KFold(n_splits = n_folds)
    
    # KFold Cross Validation
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(dataloader)):
        print("fold n°{}".format(fold_+1))
        ##Split by folder and load by dataLoader
        train_subsampler = torch.utils.data.SubsetRandomSampler(trn_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=batch_size, sampler=train_subsampler, num_workers=num_workers)
        valid_dataloader = torch.utils.data.DataLoader(dataloader, batch_size=batch_size, sampler=valid_subsampler, num_workers=num_workers)

        total_step = len(train_dataloader)
        
        writer = SummaryWriter()
        
        # Initialize Model
        model.apply(reset_weights)

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), 
                            lr=max_lr
                            )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                    max_lr, 
                                                    epochs=num_epochs,
                                                    steps_per_epoch=total_step
                                                    )

        for epoch in range(num_epochs):        
            model.train()
            train_loss = []
            train_iou  = [] 
            valid_loss = []
            valid_iou  = []           
            for batch_idx, (features,targets) in enumerate(train_dataloader):
           
                features = features.to(device)
                targets  = targets.to(device)        
                optimizer.zero_grad()

                ### FORWARD AND BACK PROP
                logits = model(features)
                cost = loss(logits, targets)            
                cost.backward()
                iou = iou_score(targets,logits).item()*100

                ### UPDATE MODEL PARAMETERS
                optimizer.step()
                scheduler.step()
                ### LOGGING
                train_loss.append(cost.item())
                train_iou.append(iou)
                
                log_step = epoch * total_step + batch_idx
                
                writer.add_scalar("train/loss", cost.item(), log_step)
                writer.add_scalar("train/iou", iou, log_step)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], log_step)

                if not batch_idx % 80:
                    print ('Epoch: %03d/%03d | Batch %03d/%03d | Train Loss: %.4f | Train IoU: %.4f%% '  
                           %(epoch+1, num_epochs, batch_idx, 
                             len(train_dataloader),
                             np.mean(train_loss),
                             np.mean(train_iou))
                          )

            ##Valid
            model.eval()                
            with torch.no_grad():
                for batch_idx, (features,targets) in enumerate(valid_dataloader):

                    features = features.to(device)
                    targets  = targets.to(device)  

                    logits = model(features)
                    cost   = loss(logits, targets)
                    iou    = iou_score(targets,logits).item()*100

                    ### LOGGING
                    valid_loss.append(cost.item())
                    valid_iou.append(iou)
                    
                    log_step = epoch * total_step + batch_idx
                    
                    writer.add_scalar("val/loss", cost.item(), log_step)
                    writer.add_scalar("val/iou", iou, log_step)
                
                print('Epoch: %03d/%03d |  Valid Loss: %.4f | Valid IoU: %.4f%%' % (
                      epoch+1, num_epochs, 
                      np.mean(valid_loss),
                      np.mean(valid_iou)))
        results[fold_+1] = np.mean(valid_iou)
        torch.save(model.state_dict(), 'fold-{:d}_IoU-{:.3f}.pt'.format(fold_+1, np.mean(valid_iou)))
        

    # Print fold results
    print(f'\nK-FOLD CROSS VALIDATION RESULTS FOR {n_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')
    

