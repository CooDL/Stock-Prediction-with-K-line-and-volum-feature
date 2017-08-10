import os, sys, os.path
import numpy as np
from PIL import Image
import torch
import multiprocessing as mp

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]
IMG_FEATURE_EX = ['_d.png', '_w.png', '_m.png', '_y.png']

#==========================================================================
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#==========================================================================
def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

#==========================================================================
def lbstrip(strmp):
    return any(strmp.strip(endexs) for endexs in IMG_FEATURE_EX)

#==========================================================================
def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

#==========================================================================
def makset(dir, class_to_idx):
    images = []
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path.rsplit('_',1)[0], class_to_idx[target])
                    images.append(item)
    images = set(images)
    images = ImgRW(images)
    return images

#==========================================================================
def ImgRW(imglist):
    newimgs = []
    for imgitm in imglist:
        flrt, target = imgitm
        daynm = flrt + '_d.png'
        weknm = flrt + '_w.png'
        mthnm = flrt + '_m.png'
        yernm = flrt + '_y.png'
        newimgs.append([daynm, weknm, mthnm, yernm, target])
    return newimgs

#==========================================================================
def Img2Bats(fillist, batchsize=1, shuffle = False, droplast = False):
    if shuffle:
        fillist = np.random.permutation(fillist)
    
    for imgidx in fillist:
        batch.append(imgidx)
        if len(batch) == batchsize:
            yield Img2Arr(batch)
            batch = []

    if len(batch) > 0 and not droplast:
        yield Img2Arr(batch) 

#==========================================================================
def Img2Arr(batch, loader = default_loader):
    targets = []
    inputs = []
    for imgset in batch:
        targets.append(imgset[-1])
        daypng = np.array(loader(imgset[0]), dtype=np.float32)
        wekpng = np.array(loader(imgset[1]), dtype=np.float32)
        mthpng = np.array(loader(imgset[2]), dtype=np.float32)
        yerpng = np.array(loader(imgset[3]), dtype=np.float32)
        inputs.append([daypng, wekpng, mthpng, yerpng])
    targets = np.array(targets, dtype=np.float32)
    inputs = torch.from_numpy(np.array(inputs))
    return inputs, targets

#==========================================================================
def SingleImg(filnm, loader= default_loader):
    tartmp = []
    ipttmp = []
    tartmp.append(filnm[-1])
    daypng = np.array(loader(filnm[0]),dtype=np.float32)
    wekpng = np.array(loader(filnm[1]),dtype=np.float32)
    mthpng = np.array(loader(filnm[2]),dtype=np.float32)
    yerpng = np.array(loader(filnm[3]),dtype=np.float32)
    ipttmp = [daypng, wekpng, mthpng, yerpng]
    return ipttmp, tartmp

#==========================================================================
class DT_Batchs(object):

    def __init__(self, root, batch_size=1, loader= default_loader, drop_last=True, shuffle = False):
        classes, class_to_idx = find_classes(root)
        imgs = makset(root, class_to_idx)
        self.root = root
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.loader = loader
        self.imgset = imgs

    def __iter__(self):
            batch = []
            for idx in self.imgset:
                    batch.append(idx)
                    if len(batch) == self.batch_size:
                            yield Img2Arr(batch, self.loader)
                            batch = []
            if len(batch) > 0 and not self.drop_last:
                    yield Img2Arr(batch, self.loader)

    def __len__(self):
            if self.drop_last:
                    return len(self.imgset) // self.batch_size
            else:
                    return (len(self.imgset) + self.batch_size - 1) // self.batch_size
#==========================================================================


#==========================================================================

#==========================================================================

#==========================================================================
