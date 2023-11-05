import torch
from loss import compute_loss_adaptive_ratio
import numpy as np
from skimage import filters
from metrics import dc, assd, precision, hd


def threshold_mask(raw_output, threshold):
    if threshold == 0:
        try:
            threshold = filters.threshold_otsu(raw_output)
        except:
            threshold = 0.5

    raw_output[raw_output > threshold] = 1
    raw_output[raw_output < 1] = 0
    return raw_output


def test(args, model, val_loader):
    model.eval()
    val_loss = 0
    acc = 0.0
    device = args.device
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += compute_loss_adaptive_ratio(output, target)
            #---------------------------------------------------------------
            output = threshold_mask(output, args.thresh_level)
            acc += precision(output, target)
            dice_arr = np.zeros((len(val_loader)))
            if args.compute_dice and batch_idx % args.log_interval == 0:
                dice_arr[batch_idx] = dc(output, target)
                print('\tdice: {:.6f}'.format(dice_arr[batch_idx]))

    val_loss /= len(val_loader)
    acc = acc / (batch_idx + 1)
    print('\ntest set: Precision: {:.4f}%'.format(acc*100))
    print('test set: Average loss: {:.6f} \n'.format(val_loss))
    return val_loss


def validate(args, model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            val_loss += compute_loss_adaptive_ratio(output, target)

    val_loss /= len(val_loader)
    print('\nValidation set: Average loss: {:.4f} \n'.format(val_loss))
    return val_loss
    

def test_dice(args, model, val_loader):
    model.eval()
    val_loss = 0
    acc_batches = []
    device = args.device
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = threshold_mask(output, args.thresh_level)
            acc = precision(output, target) * data.shape[0]
            acc_batches.append(acc)

    val_loss /= len(val_loader)
    acc = acc / (batch_idx + 1)
    print('\ntest set: Precision: {:.4f}%'.format(acc*100))
    print('test set: Average loss: {:.4f} \n'.format(val_loss))
    return val_loss


def testing_luna_dataset3d(args, model, test_set):
    model.eval()
    accs_dc = []
    accs_hd = []
    accs_assd = []
    with torch.no_grad():
        for img3d_idx in range(len(test_set)):
            imgs, masks = test_set[img3d_idx]
            pred_all_slices = []
            b = args.testing_batch_size
            img_2dslice_num = imgs.shape[0]
            for batch_start_idx in range(0, img_2dslice_num, b):
                img_batch = imgs[batch_start_idx: batch_start_idx + b]
                pred = model(img_batch)
                pred_binary = threshold_mask(pred, args.thresh_level)
                pred_all_slices.append(pred_binary)
            outputs_3d = torch.cat(pred_all_slices, dim=0)
            acc_dc = dc(outputs_3d, masks)
            accs_dc.append(acc_dc)

            acc_hd = hd(outputs_3d.cpu().numpy(), masks.cpu().numpy())
            accs_hd.append(acc_hd)

            acc_assd = assd(outputs_3d.cpu().numpy(), masks.cpu().numpy())
            accs_assd.append(acc_assd)

            print("%d dice: %06f  hd: %06f  assd:%06f" % (img3d_idx, acc_dc, acc_hd, acc_assd))
        mean_acc_dc = sum(accs_dc) / len(accs_dc)
        mean_acc_hd = sum(accs_hd) / len(accs_hd)
        accs_assd = sum(accs_assd) / len(accs_assd)
    return mean_acc_dc, mean_acc_hd, accs_assd
