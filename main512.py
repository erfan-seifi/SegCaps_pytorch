from easydict import EasyDict
import os
import torch.optim as optim
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from dataset import LunaDataset
from network_segcaps_unet512 import CapsNetU512
from trainer_segcaps import train
from test_helper import test


args = EasyDict({'epochs': 8,
                 'batch_size': 8,
                 'testing_batch_size': 32,
                 'TRAIN': True,
                 'TEST': False,
                 'LOAD_CHECKPOINT': False,
                 'compute_dice': True,
                 'lr': 0.0003,
                 'gamma': 0.4,
                 'thresh_level': 0.5,
                 'log_interval': 60,
                 'train_folder': './data/2dSlice/training_paths.txt',
                 'val_folder': './data/2dSlice/validation_paths.txt',
                 'checkpoint_path': './checkpoints/R1/epoch7.pt',
                 'save_dst': './checkpoints/R3/',
                 'device': "cuda" if torch.cuda.is_available() else "cpu"})

model = CapsNetU512()
model = nn.DataParallel(model)
start_epoch = 0


if args.LOAD_CHECKPOINT:
    print('==> Resuming from checkpoint..')
    print(args.checkpoint_path)
    model.load_state_dict(torch.load(args.checkpoint_path))
    start_epoch = 8

model.to(args.device)

if args.TRAIN:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=6, gamma=args.gamma)

    train_dataset = LunaDataset(args.train_folder, train=True)
    val_dataset = LunaDataset(args.val_folder, train=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                               num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.testing_batch_size, shuffle=False,
                                             pin_memory=True, num_workers=8)
    for epoch in range(start_epoch, args.epochs):
        # train for one epoch
        train(args, model, train_loader, optimizer, epoch)
        # evaluate on validation set
        val_loss = test(args, model, val_loader)

        scheduler.step()
        torch.save(model.state_dict(), os.path.join(args.save_dst + "epoch" + str(epoch) + ".pt"))


if args.TEST:
    from dataset import LunaDataset3D
    from test_helper import testing_luna_dataset3d

    path = './data/2dSlice/testing/'
    test_set = LunaDataset3D(path)
    mean_acc_dc, mean_acc_hd, mean_acc_assd = testing_luna_dataset3d(args, model, test_set)
    print("dice: %06f" % mean_acc_dc)
    print("hd: %06f" % mean_acc_hd)
    print("assd: %06f" % mean_acc_assd)
