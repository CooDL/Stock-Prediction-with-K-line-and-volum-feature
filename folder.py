import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import numpy as np
import multiprocessing as mp

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]

IMG_FEATURE_EX = ['_d.png', '_w.png', '_m.png', '_y.png']

#====================================
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

#====================================
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
                    item = (path.rsplit('_', 1)[0], class_to_idx[target])
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

#====================================
def default_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
#    return Image.open(path).convert('L')

#==========================================================================
def Img2Arr(path, loader = default_loader):
    inputs = []
#    print(path)
    daypng = np.array(loader(path[0]))#d_png
    wekpng = np.array(loader(path[1]))#w_png
    mthpng = np.array(loader(path[2]))#m_png
    yerpng = np.array(loader(path[3]))#y_png ,dtype=np.float32)
    inputs = [daypng, wekpng, mthpng, yerpng]
    inputs = np.array(inputs)
    return inputs

#====================================
class ImageFolder(data.Dataset):
    #====================================
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = makset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
    #====================================
    def __getitem__(self, index):
        path, target = self.imgs[index][0:-1], self.imgs[index][-1]
#       pool = mp.Pool(processes=4)
#       img = pool.apply_async(Img2Arr,(path, self.loader))
        img = Img2Arr(path, loader=self.loader)
#       pool.close()
#       pool.join()
        img = torch.from_numpy(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    #====================================
    def __len__(self):
        return len(self.imgs)
