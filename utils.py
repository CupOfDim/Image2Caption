from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import json
import torch


class CaptionDatasets(Dataset):
    def __init__(self, data_folder, split, transform=None):
        with open(data_folder, 'r') as f:
            data = json.load(f)

        self.split = split
        self.imgs = [self.read_images(path) for path in data['image_paths']]
        self.cpi = 5
        self.captions = data['tokens']
        self.caplens = data['lens']
        self.transform = transform
        self.datasize = len(self.captions) * self.cpi

    def __getitem__(self, i):
        img_idx = i // self.cpi
        cap_idx = i % self.cpi
        img = torch.FloatTensor(self.imgs[img_idx] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caption = torch.LongTensor(self.captions[img_idx][cap_idx])
        caplen = torch.LongTensor([self.caplens[img_idx][cap_idx]])

        if self.split == 'TRAIN':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[img_idx])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.datasize

    def read_images(self, path):
        img = Image.open(path)
        img = np.array(img)
        img = img.transpose(2,0,1)
        return img


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_folder, data_name, epoch, not_improve_epoches, encoder, decoder, encoder_optimizer, decoder_optimizer, bleu4, is_best):
    state = {
        'epoch':epoch,
        'not_improve_epoches':not_improve_epoches,
        'encoder':encoder,
        'decoder':decoder,
        'encoder_optimizer':encoder_optimizer,
        'decoder_optimizer':decoder_optimizer,
        'bleu4':bleu4
    }
    file_path = f'{data_folder}/checkpoint_{data_name}.pth.tar'
    torch.save(state, file_path)
    if is_best:
        torch.save(state, f'{file_path[:37]}BEST_{file_path[37:]}')


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += self.val*n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * factor