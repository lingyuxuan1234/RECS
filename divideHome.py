import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader

from data_list import *
import random, pdb, math, copy
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import pickle

import time
import shutil
import torch.nn.functional as F

from loss_funcs import LMMDLoss, MMDLoss
from loss_funcs.weight_lmmd import weight_lmmd


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(args, optimizer, iter_num, max_iter):
    decay = (1 + args.lr_gamma * iter_num / max_iter) ** (-args.lr_power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-4
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        # transforms.RandomCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])


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


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsets["target"] = ImageList_path(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dsets["test"] = ImageList_path(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def data_load_split(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_source = open(args.source_txt).readlines()
    txt_target = open(args.target_txt).readlines()

    dsets["source"] = ImageList_split(txt_source, transform=image_train())
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dsets["target"] = ImageList_split(txt_target, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)

    return dset_loaders


def gmm(all_fea, pi, mu, all_output):
    Cov = []
    dist = []
    log_probs = []

    for i in range(len(mu)):
        temp = all_fea - mu[i]
        predi = all_output[:, i].unsqueeze(dim=-1)
        Covi = torch.matmul(temp.t(), temp * predi.expand_as(temp)) / (predi.sum()) + args.epsilon * torch.eye(
            temp.shape[1]).cuda()
        try:
            chol = torch.linalg.cholesky(Covi)
        except RuntimeError:
            Covi += args.epsilon * torch.eye(temp.shape[1]).cuda() * 100
            chol = torch.linalg.cholesky(Covi)
        chol_inv = torch.inverse(chol)
        Covi_inv = torch.matmul(chol_inv.t(), chol_inv)
        logdet = torch.logdet(Covi)
        mah_dist = (torch.matmul(temp, Covi_inv) * temp).sum(dim=1)
        log_prob = -0.5 * (Covi.shape[0] * np.log(2 * math.pi) + logdet + mah_dist) + torch.log(pi)[i]
        Cov.append(Covi)
        log_probs.append(log_prob)
        dist.append(mah_dist)
    Cov = torch.stack(Cov, dim=0)
    dist = torch.stack(dist, dim=0).t()
    log_probs = torch.stack(log_probs, dim=0).t()
    zz = log_probs - torch.logsumexp(log_probs, dim=1, keepdim=True).expand_as(log_probs)
    gamma = torch.exp(zz)

    return zz, gamma


# dset_loaders["test"], netF, netB, netC, args, cnt

def evaluation(loader, netF, netB, netC, args, cnt):
    start_test = True
    iter_test = iter(loader)
    for _ in range(len(loader)):
        data = next(iter_test)
        inputs = data[0]
        labels = data[1].cuda()
        inputs = inputs.cuda()
        feas = netB(netF(inputs))
        outputs = netC(feas)
        if start_test:
            all_fea = feas.float()
            all_output = outputs.float()
            all_label = labels.float()
            start_test = False
        else:
            all_fea = torch.cat((all_fea, feas.float()), 0)
            all_output = torch.cat((all_output, outputs.float()), 0)
            all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy_return = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).data.item()

    if args.dset == 'VISDA-C':
        matrix = confusion_matrix(all_label.cpu().numpy(), torch.squeeze(predict).float().cpu().numpy())
        acc_return = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc_return.mean()
        aa = [str(np.round(i, 2)) for i in acc_return]
        acc_return = ' '.join(aa)

    all_output_logit = all_output
    all_output = nn.Softmax(dim=1)(all_output)
    all_fea_orig = all_fea
    ent = torch.sum(-all_output * torch.log(all_output + args.epsilon2), dim=1)
    unknown_weight = 1 - ent / np.log(args.class_num)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    if args.distance == 'cosine':
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float()
    K = all_output.shape[1]
    aff = all_output.float()
    # initc = torch.matmul(aff.t(), (all_fea))
    # initc = initc / (1e-8 + aff.sum(dim=0)[:, None])

    if args.pickle and (cnt == 0):
        data = {
            'all_fea': all_fea,
            'all_output': all_output,
            'all_label': all_label,
            'all_fea_orig': all_fea_orig,
        }
        filename = osp.join(args.output_dir, 'data_{}'.format(args.names[args.t]) + '.pickle')
        with open(filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        print('data_{}.pickle finished\n'.format(args.names[args.t]))

    ############################## Gaussian Mixture Modeling #############################

    uniform = torch.ones(len(all_fea), args.class_num) / args.class_num
    uniform = uniform.cuda()

    pi = all_output.sum(dim=0)
    mu = torch.matmul(all_output.t(), (all_fea))
    mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

    # zz: 所有样本的类别概率
    # gamma也是软标签，内容和zz一样
    zz, gamma = gmm((all_fea), pi, mu, uniform)
    pred_label = gamma.argmax(dim=1)

    for round in range(1):
        pi = gamma.sum(dim=0)
        mu = torch.matmul(gamma.t(), (all_fea))
        mu = mu / pi.unsqueeze(dim=-1).expand_as(mu)

        zz, gamma = gmm((all_fea), pi, mu, gamma)
        pred_label = gamma.argmax(axis=1)

    aff = gamma

    # acc时高斯混合模型给出的正确率
    acc = (pred_label == all_label).float().mean()
    # accuracy时模型给出正确率
    log_str = 'Model Prediction : Accuracy = {:.2f}%'.format(accuracy * 100) + '\n'

    if args.dset == 'VISDA-C':
        log_str += 'VISDA-C classwise accuracy : {:.2f}%\n{}'.format(aacc, acc_return) + '\n'

    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str)

    ############################## Computing JMDS score #############################

    # shape: all_num * K
    sort_zz = zz.sort(dim=1, descending=True)[0]
    zz_sub = sort_zz[:, 0] - sort_zz[:, 1]

    # 高斯模型给出的score
    LPG = zz_sub / zz_sub.max()

    if args.coeff == 'JMDS':
        PPL = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
        JMDS = (LPG * PPL)
    elif args.coeff == 'PPL':
        JMDS = all_output.gather(1, pred_label.unsqueeze(dim=1)).squeeze()
    elif args.coeff == 'NO':
        JMDS = torch.ones_like(LPG)
    else:
        JMDS = LPG

    sample_weight = JMDS

    if args.dset == 'VISDA-C':
        return aff, sample_weight, aacc / 100
    return aff, sample_weight, accuracy


def KLLoss(input_, target_, coeff, args):
    softmax = nn.Softmax(dim=1)(input_)
    # 输出结果是横着的
    kl_loss = (- target_ * torch.log(softmax + args.epsilon2)).sum(dim=1)
    kl_loss *= coeff
    return kl_loss.mean(dim=0)


# (args, netF, netB, netC, soft_pseudo_label, coeff,
#                                  dset_loaders["test"], ratio=0.5, alpha=0.0)
def split_images_perclass_v3(args, soft_pseudo_label, coeff, test_loader, ratio=0.5):
    all_path = []

    for batch_idx, (images, label, index, path) in enumerate(test_loader):
        for img_path in path:
            all_path.append(img_path)

    all_pred = soft_pseudo_label.argmax(dim=1)

    expect_num = soft_pseudo_label.shape[0] / args.class_num
    min_num, max_num = int(expect_num * (ratio - 0.2)), int(expect_num * (ratio + 0.2))
    print(expect_num, min_num, max_num)

    for i in range(args.class_num):
        now_index = torch.where(all_pred == i)[0]
        #
        limit = int(len(coeff[now_index]) * ratio)
        limit = max(limit, min_num)
        limit = min(limit, max_num)
        sim_index_i = coeff[now_index].argsort(descending=True)[:limit]
        sim_index_i = now_index[sim_index_i]
        if i == 0:
            sim_index = sim_index_i.detach().cpu().long()
        else:
            sim_index = torch.cat((sim_index, sim_index_i.detach().cpu().long()), 0)

    source_txt = open('./data/source.txt', 'w')
    target_txt = open('./data/target.txt', 'w')
    for batch_idx, img_path in enumerate(all_path):
        idx = torch.where(sim_index == batch_idx)[0]
        if idx.numel() != 0:
            save_idx = sim_index[idx]
            source_info = img_path + ' ' + str(all_pred[save_idx].item()) + ' ' + str(batch_idx)
            source_txt.write(source_info + '\n')
            source_txt.flush()
        else:
            target_info = img_path + ' ' + str(all_pred[batch_idx].item()) + ' ' + str(batch_idx)
            target_txt.write(target_info + '\n')
            target_txt.flush()


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def guassian_kernel(source, target, kernel_mul, kernel_num, fix_sigma):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i)
                      for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


def train_target(args):
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()
    elif args.net[0:3] == 'vgg':
        netF = network.VGGBase(vgg_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    ####################################################################
    modelpath = args.output_dir_src + '/ckpt_F.pt'
    print('modelpath: {}'.format(modelpath))
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/ckpt_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/ckpt_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group.append({'params': netF.parameters(), 'lr': args.lr * 0.1})
    param_group.append({'params': netB.parameters(), 'lr': args.lr * 1.0})
    param_group.append({'params': netC.parameters(), 'lr': args.lr * 0.1})

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    cnt = 0
    acc_init = 0

    dset_loaders = data_load(args)

    epochs = []
    accuracies = []

    augment1 = transforms.Compose([
        # transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
    ])

    transfer_loss = weight_lmmd(args.class_num)
    cross_entropy = nn.CrossEntropyLoss()

    iter_each_epoch = len(dset_loaders["test"]) / 2
    max_iter = args.max_epoch * len(dset_loaders["test"]) / 2

    start_time = time.time()
    for ep in range(args.max_epoch):
        netF.eval()
        netB.eval()
        netC.eval()
        with torch.no_grad():
            # Compute JMDS score at offline & evaluation.
            soft_pseudo_label, coeff, accuracy = evaluation(
                dset_loaders["test"], netF, netB, netC, args, cnt
            )

            # 根据置信度划分样本，并保存对应的标签
            split_images_perclass_v3(args, soft_pseudo_label, coeff, dset_loaders["test"], ratio=0.4)

            epochs.append(cnt)
            accuracies.append(np.round(accuracy * 100, 2))

        if accuracy >= acc_init:
            acc_init = accuracy
            torch.save(netF.state_dict(), osp.join(args.output_dir, 'ckpt_F.pt'))
            torch.save(netB.state_dict(), osp.join(args.output_dir, 'ckpt_B.pt'))
            torch.save(netC.state_dict(), osp.join(args.output_dir, 'ckpt_C.pt'))

        cnt += 1
        args.source_txt = './data/source.txt'
        args.target_txt = './data/target.txt'
        split_data = data_load_split(args)

        netF.train()
        netB.train()
        netC.train()

        for batch_idx, ((src_images, src_label, src_index), (tgt_images, tgt_label, tgt_index)) in enumerate(
                zip(split_data["source"], split_data["target"])):
            current_iter = ep * iter_each_epoch + batch_idx
            lr_scheduler(args, optimizer, iter_num=current_iter, max_iter=max_iter)

            # 三个损失
            # 源域和目标域对齐
            # 互信息
            src_images = augment1(src_images)
            tgt_images = augment1(tgt_images)
            src_images, src_label, tgt_images = src_images.cuda(), src_label.cuda(), tgt_images.cuda()

            src_fea = netB(netF(src_images))
            tgt_fea = netB(netF(tgt_images))

            src_output = netC(src_fea)
            tgt_output = netC(tgt_fea)

            src_logit = nn.Softmax(dim=1)(src_output)
            tgt_logit = nn.Softmax(dim=1)(tgt_output)

            # 源域上的cross entropy
            loss_src = cross_entropy(src_output, src_label)

            # 目标域上的鲁棒损失
            # loss_robust = reverse_KLLoss(tgt_output, soft_pseudo_label[tgt_index], coeff[tgt_index], args)

            # 源域和目标域对齐
            adapt_loss = transfer_loss(src_fea, tgt_fea, src_label, tgt_logit, coeff[src_index], coeff[tgt_index])

            total_loss = loss_src + 1.0 * adapt_loss
            # log_dir = 'loss_src:{:.2f}  adapt_loss:{:.2f} '.format(loss_src.item(), adapt_loss.item())
            # print(log_dir)
            #
            # print(log_dir)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("程序运行时间: {:.4f} 秒".format(elapsed_time))

    ####################################################################

    # if args.issave:
    #     torch.save(best_netF, osp.join(args.output_dir, 'ckpt_F.pt'))
    #     torch.save(best_netB, osp.join(args.output_dir, 'ckpt_B.pt'))
    #     torch.save(best_netC, osp.join(args.output_dir, 'ckpt_C.pt'))

    log_str = '\nAccuracies history : {}\n'.format(accuracies)
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(epochs, accuracies, 'o-')
    # plt.savefig(osp.join(args.output_dir, 'png_{}.png'.format(args.prefix)))
    # plt.close()

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SHOT')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")

    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--batch_size', type=int, default=32, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech', 'DomainNet'])
    parser.add_argument('--net', type=str, default='resnet50', help="alexnet, vgg16, resnet50, res101")
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--warm', type=float, default=0.0)
    parser.add_argument('--coeff', type=str, default='JMDS', choices=['LPG', 'JMDS', 'PPL', 'NO'])
    parser.add_argument('--pickle', default=False, action='store_true')
    parser.add_argument('--lr_power', type=float, default=0.75)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-6)
    parser.add_argument('--epsilon2', type=float, default=1e-6)
    parser.add_argument('--delta', type=float, default=2.0)
    parser.add_argument('--n_power', type=int, default=1)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--da', type=str, default='uda', choices=['uda'])
    parser.add_argument('--issave', type=bool, default=True)

    # parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_gamma', type=float, default=10.0)
    parser.add_argument('--lr_decay', type=float, default=0.75)

    parser.add_argument('--max_epoch', type=int, default=20, help="max iterations")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--seed', type=int, default=3, help="random seed")
    parser.add_argument('--output', type=str, default='ckps/time33/')
    parser.add_argument('--output_src', type=str, default='ckps/time3/')
    parser.add_argument('--s', type=int, default=0, help="source")

    args = parser.parse_args()

    if args.dset == 'office-home':
        args.names = ['Art', 'Clipart', 'Product', 'RealWorld']
        # args.names = ['Art', 'Product', 'RealWorld', 'Clipart']
        args.class_num = 65
    if args.dset == 'office':
        args.names = ['amazon', 'webcam', 'dslr']
        # args.names = ['webcam', 'dslr', 'amazon']
        args.class_num = 31
    if args.dset == 'VISDA-C':
        args.names = ['train', 'validation']
        args.class_num = 12
    if args.dset == 'office-caltech':
        args.names = ['amazon', 'caltech', 'dslr', 'webcam']
        args.class_num = 10
    if args.dset == 'DomainNet':
        args.names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.class_num = 345

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed

    ############# If you want to obtain the stochastic result, comment following lines. #############
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)

    for i in range(len(args.names)):
        start = time.time()
        if i == args.s:
            continue
        args.t = i

        folder = '/home/suncy/'
        # folder = '/opt/data/private/users/suncy/'
        args.s_dset_path = folder + args.dset + '/' + args.names[args.s] + '.txt'
        args.t_dset_path = folder + args.dset + '/' + args.names[args.t] + '.txt'
        args.test_dset_path = folder + args.dset + '/' + args.names[args.t] + '.txt'

        args.name = args.names[args.s][0].upper() + args.names[args.t][0].upper()

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, args.name)
        args.output_dir = osp.join(args.output, args.da, args.dset, args.name)

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        ####################################################################
        if not osp.exists(osp.join(args.output_dir, 'ckpt_F.pt')):
            args.out_file = open(osp.join(args.output_dir, 'log.txt'), 'w')
            args.out_file.write(print_args(args) + '\n')
            args.out_file.flush()
            train_target(args)

            total_time = time.time() - start
            log_str = 'Consumed time : {} h {} m {}s'.format(total_time // 3600, (total_time // 60) % 60,
                                                             np.round(total_time % 60, 2))
            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str)
        else:
            print('{} Already exists'.format(osp.join(args.output_dir, 'log.txt')))
