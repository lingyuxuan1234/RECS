from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms

import network
import torch
import torch.nn as nn

from data_list import ImageList_idx


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])


# t为当前数据集的第几个任务
def data_load(dset, t):
    if dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        class_num = 65
    if dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        class_num = 31
    if dset == 'VISDA-C':
        names = ['train', 'validation']
        class_num = 12
    dsets = {}
    dset_loaders = {}
    train_bs = 64

    folder = '/home/suncy/'
    test_dset_path = folder + 'office-home1/PA.txt'

    txt_test = open(test_dset_path).readlines()

    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=4,
                                      drop_last=False)

    return dset_loaders


def draw_data(data, label, name=''):
    X1 = TSNE(n_components=2).fit_transform(data.cpu().numpy())
    Y1 = label.cpu().numpy()
    plt.cla()
    plt.scatter(X1[:, 0], X1[:, 1], 10, label)
    plt.savefig('img' + name + '.pdf')
    return


# T-SNE

# 65 resnet50   /home/suncy/DSFUDA/ckps/source/uda/office-home/R
def test(class_num, net, output_dir_src):
    netF = network.ResBase(res_name=net).cuda()
    netB = network.feat_bottleneck(type="bn", feature_dim=netF.in_features, bottleneck_dim=256).cuda()
    netC = network.feat_classifier(type="wn", class_num=class_num, bottleneck_dim=256).cuda()

    modelpath = output_dir_src + '/ckpt_F.pt'
    print('modelpath: {}'.format(modelpath))
    netF.load_state_dict(torch.load(modelpath))
    modelpath = output_dir_src + '/ckpt_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = output_dir_src + '/ckpt_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    netF.eval()
    netB.eval()
    netC.eval()

    loader = data_load("office-home", 0)

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader["test"])):
            data = iter_test.next()
            inputs = data[0]
            inputs = inputs.cuda()
            feature = netB(netF(inputs))
            outputs = netC(feature)
            if start_test:
                all_fea = feature.float().cpu()
                all_output = outputs.float().cpu()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feature.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    draw_data(all_fea, predict, name='PA_src')


test(65, 'resnet50', '/home/suncy/NCAA/ckps/src3/uda/office-home/P')

# nohup python3 utils.py  > TSNE.txt 2>&1 &
