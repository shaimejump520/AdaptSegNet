import numpy as np
import torch 
import torch.utils.data
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps

def roundrobin_infinite(*loaders):
    if not loaders:
        return
    iters = [iter(loader) for loader in loaders]
    while True:
        for i in range(len(iters)):
            it = iters[i]
            try:
                yield next(it)
            except StopIteration:
                iters[i] = iter(loaders[i])
                yield next(iters[i])

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n**2).reshape(n, n)

def result_stats(hist):
    acc_overall = np.diag(hist).sum() / (hist.sum() + 1e-8)
    acc_percls = np.diag(hist) / (hist.sum(1) + 1e-8)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8)
    freq = hist.sum(1) / hist.sum()
    fwIU = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc_overall, acc_percls, iu, fwIU

def aug_batch_tensor(datas, policies, rand_p1, rand_p2, random_dir):
    aug_datas = []
    for i in range(datas.shape[0]):
        aug_data = policies[i](transforms.ToPILImage()(datas[i].cpu()).convert('RGB'), rand_p1[i], rand_p2[i], random_dir[i])
        aug_datas.append(transforms.ToTensor()(aug_data))
    aug_datas = torch.stack(aug_datas)
    
    return aug_datas

def aug_batch_numpy(datas, policies, rand_p1, rand_p2, random_dir):
    skip_policy = ['Invert', 'Sharpness', 'AutoContrast', 'Posterize',
                   'Equalize', 'Contrast', 'Color', 'Solarize', 'Brightness']
    aug_datas = []
    for i in range(datas.shape[0]):
        if policies[i]._operation1_name in skip_policy:
            if policies[i]._operation2_name in skip_policy:
                aug_data = datas[i]
            else:
                aug_data = policies[i](Image.fromarray(datas[i]).convert('RGB'), 
                                       1, rand_p2[i], random_dir[i])
                aug_data = np.asarray(aug_data.convert('L'), np.float32)
        elif policies[i]._operation2_name in skip_policy:
            aug_data = policies[i](Image.fromarray(datas[i]).convert('RGB'), 
                                   rand_p1[i], 1, random_dir[i])
            aug_data = np.asarray(aug_data.convert('L'), np.float32)
        else:
            aug_data = policies[i](Image.fromarray(datas[i]).convert('RGB'), 
                                   rand_p1[i], rand_p2[i], random_dir[i])
            aug_data = np.asarray(aug_data.convert('L'), np.float32)
        aug_datas.append(torch.from_numpy(aug_data))
    aug_datas = torch.stack(aug_datas)
    
    return aug_datas

def transform_batch(datas, transforms):
    trans_datas = []
    for i in range(datas.shape[0]):
        trans_data = transforms(datas[i])
        trans_datas.append(trans_data)
    trans_datas = torch.stack(trans_datas)
    
    return trans_datas

def aug_batch_likelihood_test(datas):
    aug_datas = []
    for i in range(datas.shape[0]):
        aug_data_c_s = []
        for j in range(datas.shape[1]):
            aug_data_c = transforms.ToPILImage()(datas[i][j]).convert('RGB')
            aug_data_c_s.append(transforms.ToTensor()(aug_data_c)[0])
        aug_data_c_s = torch.stack(aug_data_c_s)
        aug_datas.append(aug_data_c_s)
    aug_datas = torch.stack(aug_datas)
    
    return aug_datas

def aug_batch_likelihood(datas, policies, rand_p1, rand_p2, random_dir):
    aug_datas = []
    for i in range(datas.shape[0]):
        aug_data_c_s = []
        for j in range(datas.shape[1]):
            aug_data_c = policies[i](transforms.ToPILImage()(datas[i][j].cpu()).convert('RGB'), rand_p1[i], rand_p2[i], random_dir[i])
            aug_data_c_s.append(transforms.ToTensor()(aug_data_c)[0])
        aug_data_c_s = torch.stack(aug_data_c_s)
        aug_datas.append(aug_data_c_s)
    aug_datas = torch.stack(aug_datas)
    
    return aug_datas
