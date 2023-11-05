from loss import compute_loss, compute_loss_adaptive_ratio, compute_loss_balanced
import time
from utils import AverageMeter


# unzip -u "/content/drive/My Drive/SegCaps/training.zip" -d "/content/drive/My Drive/SegCaps/training"
def train(args, model, train_loader, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    model.train()
    model = model.to(args.device)
    total_loss = 0
    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        optimizer.zero_grad()
        data, target = data.to(args.device), target.to(args.device)
        output = model(data)

        loss = compute_loss_adaptive_ratio(output, target)
        total_loss += loss.detach().clone()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.log_interval == 0:
            print('Epoch: [{}][{}/{}]\t'
                  'Loss: {:.6f}\t'
                  'Average Loss: {:.6f}\t'
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data: {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  epoch, batch_idx, len(train_loader),
                  loss.item(),
                  total_loss / (batch_idx+1), 
                  batch_time=batch_time,
                  data_time=data_time
                  ))

    return total_loss.item()
