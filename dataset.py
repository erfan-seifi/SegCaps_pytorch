import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from data_augmentation import RandomCrop, RandomHorizontallyFlip, RandomGaussianBlur
import torch


class LunaDataset(Dataset):
    def __init__(self, img_mask_path, train=False):
        self.train = train
        self.img_paths = []
        with open(img_mask_path) as f:
            lines = [line.rstrip() for line in f]
            for line in lines:
                self.img_paths.append(line)
        self.len_dataset = len(self.img_paths)

        self.RandomCrop = RandomCrop(512, padding=20)
        self.HorizontalFlip = RandomHorizontallyFlip()
        self.GaussianBlur = RandomGaussianBlur()
        self.colorJitter = transforms.Compose([transforms.ColorJitter(brightness=0.2, contrast=0.2)])
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('L')
        mask_path = self.img_paths[idx].replace('.bmp', '_mask.bmp')
        mask = Image.open(mask_path).convert('L')
        if self.train:
            img, mask = self.RandomCrop(img, mask)
            img, mask = self.HorizontalFlip(img, mask)
            img = self.GaussianBlur(img)
            img = self.colorJitter(img)
        img, mask = self.toTensor(img), self.toTensor(mask)
        return img, mask.squeeze()

    # #test
    # val_dataset = LunaDataset("./data/2dSlice/validation_paths.txt", train=True)
    # img, mask = val_dataset[100]
    # from matplotlib import pyplot as plt
    # f, axarr = plt.subplots(1,2)
    # axarr[0].imshow(img, cmap='gray')
    # axarr[1].imshow(mask, cmap='gray')
    # plt.show()


class LunaDataset3D(Dataset):
    def __init__(self, folder_path):
        self.img3d_paths = sorted([f.path for f in os.scandir(folder_path) if f.is_dir()])
        self.len_dataset = len(self.img3d_paths)
        self.toTensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        imgs = []
        masks = []
        print("Dataset loading ", self.img3d_paths[idx])
        x = glob.glob(os.path.join(self.img3d_paths[idx], '*.bmp'))
        x = sorted(x)
        x = [i for i in x if 'mask' not in i]
        for i in x:
            img = Image.open(i).convert('L')
            img = self.toTensor(img).unsqueeze(dim=0)  # grayscale image
            imgs.append(img)
            mask3d_path = i.replace('.bmp', '_mask.bmp')
            mask = Image.open(mask3d_path).convert('L')
            mask = self.toTensor(mask)
            masks.append(mask)
        imgs = torch.cat(imgs, dim=0)
        masks = torch.cat(masks, dim=0)
        return imgs, masks


def split_image_to_slices(mode):
    import glob
    from PIL import Image

    data_idx = 0
    path = './data/3d/' + mode
    total_3d = len(glob.glob(path + '/*.np[yz]'))
    for idx in range(total_3d):
        print("%d/%d" % (idx, total_3d))
        data_i = []
        img_path = os.path.join(path, str(idx) + '.npz')
        data_i.append(np.load(img_path, allow_pickle=True))
        data_i[0] = data_i[0]['arr_0'].item()
        idx = "%03d" % idx
        for j in range(data_i[0]['img'].shape[-1]):
            total_slices = data_i[0]['img'].shape[-1]
            slice_idx = j
            data_i_sliced = {}
            data_i_sliced['img'] = data_i[0]['img'][:, :, slice_idx]
            data_i_sliced['mask'] = data_i[0]['mask'][:, :, slice_idx]
            data_i_sliced['name'] = data_i[0]['name']
            data_i_sliced['idx'] = idx
            data_i_sliced['slice'] = slice_idx

            # for debug
            # from matplotlib import pyplot as plt
            # if j == 120:
                # plt.imshow(data_i_sliced['img'], cmap='gray')
                # plt.show()
                # print(data_i_sliced['img'])
                # plt.imshow(data_i_sliced['mask'], cmap='gray')
                # plt.show()
                # print(data_i_sliced['mask'])

            slice_idx = "%03d" % slice_idx
            im = data_i_sliced['img']
            minp, maxp = 0.250, 0.252
            assert (np.sum(im < minp) == 0)
            assert (np.sum(im > maxp) == 0)
            im = (im - minp) / (maxp - minp)
            im = (im * 255).astype(np.uint8)
            folder = './data/2dSlice/' + mode + '/' + str(idx)
            im = Image.fromarray(im).convert('RGB')
            if not os.path.isdir(folder):
                os.mkdir(folder)
            img_path = folder + '/slice_' + str(slice_idx) + '.bmp'
            im.save(img_path)

            mask = data_i_sliced['mask']
            assert (np.sum(mask < 0) == 0)
            assert (np.sum(mask > 1) == 0)
            assert ((np.sum(mask == 1) + np.sum(mask == 0)) == 512 * 512)
            mask[mask == 1] = 255
            mask_path = folder + '/slice_' + str(slice_idx) + '_mask.bmp'
            mask = Image.fromarray(mask).convert('RGB')
            mask.save(mask_path)
    return


def generate_img_mask_path_txt(mode):
    import glob
    imgs = []
    for f in glob.glob('./data/2dSlice/' + mode + '/**/*.bmp', recursive=True):
        #print(f)
        if 'mask' not in f:
            imgs.append(f)
    imgs = sorted(imgs)
    with open("./data/2dSlice/" + mode + "_paths.txt", "w") as out:
        for i in imgs:
            out.write("%s\n" % i)
    return


IMAGE3D_TO_SLICES = False
if IMAGE3D_TO_SLICES:
    for mode in ['training', 'validation', 'testing']:
        split_image_to_slices(mode=mode)
        generate_img_mask_path_txt(mode=mode)