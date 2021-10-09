import argparse
import torch 
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import os
import os.path as osp

from model.CLAN_G import Res_Deeplab
from model.CLAN_D import FCDiscriminator

from utils.loss import CrossEntropy2d
from utils.loss import WeightedBCEWithLogitsLoss
from model.cond_model import  integrated_model


from dataset.gta5_dataset import GTA5DataSet
from dataset.synthia_dataset import SYNTHIADataSet
#from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.imagenet_dataset import imagenetDataset

from pytorch_AdaIN.net2 import decoder,vgg
from pytorch_AdaIN.function import adaptive_instance_normalization,calc_mean_std
import time

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'ResNet'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
IGNORE_LABEL = 255

MOMENTUM = 0.9
NUM_CLASSES = 19
RESTORE_FROM = './model/DeepLab_resnet_pretrained_init-f81d91e8.pth'

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 2000
SNAPSHOT_DIR = './snapshots/decoder_disc_org_cond_alt_v2_gta_imagenet_num_workers'

#Hyper Paramters
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 2.5e-4
LEARNING_RATE_D = 1e-4
NUM_STEPS = 100000
NUM_STEPS_STOP = 100000  # Use damping instead of early stopping
PREHEAT_STEPS = int(NUM_STEPS_STOP/20)
POWER = 0.9
RANDOM_SEED = 1234

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
#torch.backends.cudnn.deterministic = True

SOURCE = 'GTA5' 
TARGET = 'cityscapes'
SET = 'train'
if SOURCE == 'GTA5':
    INPUT_SIZE_SOURCE = '1280,720'#1280,720 #640,360, 960,540
    DATA_DIRECTORY = '/home/gabriel/data/gta5/gta5'
    DATA_LIST_PATH = './dataset/gta5_list/train.txt'
    Lambda_weight = 0.01
    Lambda_adv = 0.001
    Lambda_local = 40
    Epsilon = 0.4
elif SOURCE == 'SYNTHIA':
    INPUT_SIZE_SOURCE = '1280,760' #1280,760
    DATA_DIRECTORY = '/home/gabriel/data/synthia/RAND_CITYSCAPES'
    DATA_LIST_PATH = './dataset/synthia_list/train.txt'
    Lambda_weight = 0.01
    Lambda_adv = 0.001
    Lambda_local = 10
    Epsilon = 0.4
    
INPUT_SIZE_TARGET = '1024,512' #1024,512
DATA_DIRECTORY_TARGET = '/home/gabriel/data/cityscapes'
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
torch.autograd.set_detect_anomaly(True)

def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return torch.sum(-torch.mul(prob, torch.log2(prob + 1e-30))) / (n*c*h*w*np.log2(c))



def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : ResNet")
    parser.add_argument("--source", type=str, default=SOURCE,
                        help="available options : GTA5, SYNTHIA")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size-source", type=str, default=INPUT_SIZE_SOURCE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=LEARNING_RATE_D,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def optimize_beta( beta, bottleneck_loss):
        """
        perform a step for updating the adaptive beta
        :param beta: old value of beta
        :param bottleneck_loss: current value of bottleneck loss
        :return: beta_new => updated value of beta
        """
        # please refer to the section 4 of the vdb_paper in literature
        # for more information about this.
        # this performs gradient ascent over the beta parameter
        bottleneck_loss=bottleneck_loss.detach()
        beta_new = max(0, beta + (1e-6 * bottleneck_loss)) #alpha:1e-6
        
        # return the updated beta value:
        return beta_new


def bottleneck_loss(mus, sigmas, i_c, alpha=1e-8):
        """
        calculate the bottleneck loss for the given mus and sigmas
        :param mus: means of the gaussian distributions
        :param sigmas: stds of the gaussian distributions
        :param i_c: value of bottleneck
        :param alpha: small value for numerical stability
        :return: loss_value: scalar tensor
        """
        # add a small value to sigmas to avoid inf log
        kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                      - torch.log((sigmas ** 2) + alpha) - 1, dim=1))

        # calculate the bottleneck loss:
        bottleneck_loss = (torch.mean(kl_divergence) - i_c)

        # return the bottleneck_loss:
        return bottleneck_loss


def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d(NUM_CLASSES).cuda(gpu)
    return criterion(pred, label)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    if i_iter < PREHEAT_STEPS:
        lr = lr_warmup(args.learning_rate_D, i_iter, PREHEAT_STEPS)
    else:
        lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def weightmap(pred1, pred2):
    output = 1.0 - torch.sum((pred1 * pred2), 1).view(1, 1, pred1.size(2), pred1.size(3)) / \
    (torch.norm(pred1, 2, 1) * torch.norm(pred2, 2, 1)).view(1, 1, pred1.size(2), pred1.size(3))
    return output


def main():
    """Create the model and start the training."""

    h, w = map(int, args.input_size_source.split(','))
    input_size_source = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)

    cudnn.enabled = True
    
    # Create Network
    model = Res_Deeplab(num_classes=args.num_classes)
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split('.')
        if not args.num_classes == 19 or not i_parts[1] == 'layer5':
            new_params['.'.join(i_parts[1:])] = saved_state_dict[i]

    if args.restore_from[:4] == './mo':        
        model.load_state_dict(new_params)
    else:
        model.load_state_dict(saved_state_dict)
    
    model.train()
    model.cuda(args.gpu)

    im1=integrated_model()
    im1.train()
    cudnn.benchmark = True
    # Init D
    model_D = FCDiscriminator(num_classes=args.num_classes)
    #
# =============================================================================
#    #for retrain     
#    saved_state_dict_D = torch.load(RESTORE_FROM_D)
#    model_D.load_state_dict(saved_state_dict_D)
# =============================================================================
    
    model_D.train()
    model_D.cuda(args.gpu)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    
    if args.source == 'GTA5':
        trainloader = data.DataLoader(
            GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size_source,
                        scale=True, mirror=True, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = data.DataLoader(
            SYNTHIADataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                        crop_size=input_size_source,
                        scale=True, mirror=True, mean=IMG_MEAN),
            batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    styleloader=data.DataLoader(imagenetDataset(max_iters=args.num_steps * args.iter_size * args.batch_size,crop_size=input_size_source,
                        scale=True, mirror=True, mean=IMG_MEAN), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    styleloader_iter = enumerate(styleloader)


    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))

    optimizer_D.zero_grad()
    optimizer_im = optim.SGD(list(im1.parameters()),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    optimizer_im.zero_grad()


    bce_loss = torch.nn.BCEWithLogitsLoss()
    weighted_bce_loss = WeightedBCEWithLogitsLoss()

    interp_source = nn.Upsample(size=(input_size_source[1], input_size_source[0]), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    interp_target_bn = nn.Upsample(size=(int(input_size_target[1]/4), int(input_size_target[0]/4)), mode='bilinear', align_corners=True)
    
    # Labels for Adversarial Training
    source_label = 0
    target_label = 1
    print('exp = {}'.format(args.snapshot_dir))
    beta=0
    
    for i_iter in range(args.num_steps):
        time_start=time.time()

        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)

        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)


        optimizer_im.zero_grad()
        ## adjust learning rate
        adjust_learning_rate(optimizer_im, i_iter)

        damping = (1 - i_iter/NUM_STEPS)

        decoder1 = decoder
        vgg1 = vgg
        decoder1.eval()
        vgg1.eval()
        decoder1.load_state_dict(torch.load('/home/gabriel/CLAN/pytorch_AdaIN/models/decoder.pth'))
        vgg1.load_state_dict(torch.load('/home/gabriel/CLAN/pytorch_AdaIN/models/vgg_normalised.pth'))
        vgg1 = nn.Sequential(*list(vgg.children())[:31])
        im1.train()
        vgg1.cuda(args.gpu)
        decoder1.cuda(args.gpu)
        im1.cuda(args.gpu)        
        #======================================================================================
        # train G
        #======================================================================================

        #Remove Grads in D
        for param in model_D.parameters():
            param.requires_grad = False
        # Train integrated model
        for param in model.parameters():
            param.requires_grad = False
        for param in im1.parameters():
            param.requires_grad = True


        _, batch = next(trainloader_iter)
        images_s, labels_s, _, _, _ = batch
        images_s = Variable(images_s).cuda(args.gpu)
        pred_source1, pred_source2 = model(images_s)
        pred_source=F.softmax(pred_source1+pred_source2)
        #label smoothing
        a_smth=0.1	
        pred_source= (1-a_smth)*pred_source + (a_smth/args.num_classes)
        
        _, stylebatch = next(styleloader_iter)
        style_img, _, _ =stylebatch
        
        def calc_content_loss( input, target):
           interp_tgt = nn.Upsample(size=(target.size(2), target.size(3)), mode='bilinear', align_corners=True)
           input=interp_tgt(input)

           assert (input.size() == target.size())
           assert (target.requires_grad is False)
           loss=nn.MSELoss()
           return loss(input, target)

        def calc_style_loss( input, target):
           interp_tgt = nn.Upsample(size=(target.size(2), target.size(3)), mode='bilinear', align_corners=True)
           input=interp_tgt(input)
           assert (input.size() == target.size())
           assert (target.requires_grad is False)
           input_mean, input_std = calc_mean_std(input)
           target_mean, target_std = calc_mean_std(target)
           loss=nn.MSELoss()
           return loss(input_mean, target_mean) + \
               loss(input_std, target_std)
        images_style = Variable(style_img).cuda(args.gpu)
        with torch.no_grad():
           content_f = vgg1(images_s)
           style_f = vgg1(images_style)
           originalRandomNoise = torch.randn(style_f.shape)
           n_u, n_e, n_v = torch.svd(originalRandomNoise,some=True)
           n_v=n_v.reshape(style_f.shape)
           n_v=n_v.cuda(args.gpu)
           style_f*=(n_v+1)

        fused=im1(style_f,pred_source) ### change model 
        feat = adaptive_instance_normalization(content_f, fused)
        alpha=0.5
        feat1 =alpha*feat+(1-alpha)*content_f
        images_t=decoder1(feat1)
        style_loss1 = 1e-8*calc_style_loss(feat,style_f.detach())
        style_loss2 = 1e-8*calc_style_loss(feat,content_f.detach())
        content_loss= 1e-8*calc_content_loss(feat,content_f.detach())

        pred_target1, pred_target2 = model(images_t)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        weight_map = weightmap(F.softmax(pred_target1, dim = 1), F.softmax(pred_target2, dim = 1))
        t_out=model_D(F.softmax(pred_target1 + pred_target2, dim = 1))
        
        D_out=interp_target(t_out)

        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            loss_adv = weighted_bce_loss(D_out, 
                                    Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(
                                        args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_adv = bce_loss(D_out,
                          Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu))


        loss_adv1 =loss_adv * Lambda_adv * damping
        im_loss = -loss_adv1 +  style_loss1 - style_loss2 + content_loss

        # content loss - feat compare with content_f (minimize content loss)
        # style loss   - feat compared with style_f (minimize style f loss) + feat compared with content f (maximise loss)
        # loss adv maximize the adversarial loss
        im_loss.backward()

        ## Train with Source

        for param in model.parameters():
            param.requires_grad = True
        for param in im1.parameters():
            param.requires_grad = False
        pred_source1, pred_source2 = model(images_s)
        pred_source1 = interp_source(pred_source1)
        pred_source2 = interp_source(pred_source2)
        pred_source=F.softmax(pred_source1+pred_source2)
        #label smoothing
        a_smth=0.1	
        pred_source= (1-a_smth)*pred_source + (a_smth/args.num_classes)

		
        #Segmentation Loss
        loss_seg = (loss_calc(pred_source1, labels_s, args.gpu) + loss_calc(pred_source2, labels_s, args.gpu))
        loss_seg.backward()

        # Train with Stylized source images
        with torch.no_grad():
           content_f = vgg1(images_s)
           style_f = vgg1(images_style)
           originalRandomNoise = torch.randn(style_f.shape)
           n_u, n_e, n_v = torch.svd(originalRandomNoise,some=True)
           n_v=n_v.reshape(style_f.shape)
           n_v=n_v.cuda(args.gpu)
           style_f*=(n_v+1)

           fused=im1(style_f,pred_source)
           feat = adaptive_instance_normalization(content_f, fused)
           alpha=0.5
           feat1 =alpha*feat+(1-alpha)*content_f
        
           images_t=decoder1(feat1)
           del decoder1,vgg1,feat,feat1
        
        pred_target1, pred_target2 = model(images_t)
        pred_target1 = interp_target(pred_target1)
        pred_target2 = interp_target(pred_target2)

        weight_map = weightmap(F.softmax(pred_target1, dim = 1), F.softmax(pred_target2, dim = 1))
        t_out=model_D(F.softmax(pred_target1 + pred_target2, dim = 1))
        
        D_out=interp_target(t_out)
        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            loss_adv = weighted_bce_loss(D_out, 
                                    Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(
                                        args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_adv = bce_loss(D_out,
                          Variable(torch.FloatTensor(D_out.data.size()).fill_(source_label)).cuda(args.gpu))

        loss_adv2 = loss_adv * Lambda_adv * damping
        loss_adv2.backward()
        
        #Weight Discrepancy Loss
        W5 = None
        W6 = None
        if args.model == 'ResNet':

            for (w5, w6) in zip(model.layer5.parameters(), model.layer6.parameters()):
                if W5 is None and W6 is None:
                    W5 = w5.view(-1)
                    W6 = w6.view(-1)
                else:
                    W5 = torch.cat((W5, w5.view(-1)), 0)
                    W6 = torch.cat((W6, w6.view(-1)), 0)
        
        loss_weight = (torch.matmul(W5, W6) / (torch.norm(W5) * torch.norm(W6)) + 1) # +1 is for a positive loss
        loss_weight = loss_weight * Lambda_weight * damping * 2
        loss_weight.backward()
        
        #======================================================================================
        # train D
        #======================================================================================


        pred_source1 = pred_source1.detach()
        pred_source2 = pred_source2.detach()
        
        # Bring back Grads in D
        for param in model_D.parameters():
            param.requires_grad = True
            
        # Train with Source
        s_out=model_D(F.softmax(pred_source1 + pred_source2, dim = 1))

        D_out_s = interp_source(s_out)
        loss_D_s = bce_loss(D_out_s,
                          Variable(torch.FloatTensor(D_out_s.data.size()).fill_(source_label)).cuda(args.gpu))


        # Train with Target
        pred_target1 = pred_target1.detach()
        pred_target2 = pred_target2.detach()
        weight_map = weight_map.detach()
        
        t_out=model_D(F.softmax(pred_target1 + pred_target2, dim = 1))
        D_out_t =interp_target(t_out)

        #Adaptive Adversarial Loss
        if(i_iter > PREHEAT_STEPS):
            loss_D_t = weighted_bce_loss(D_out_t, 
                                    Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(
                                        args.gpu), weight_map, Epsilon, Lambda_local)
        else:
            loss_D_t = bce_loss(D_out_t,
                          Variable(torch.FloatTensor(D_out_t.data.size()).fill_(target_label)).cuda(args.gpu))

        sum_loss=loss_D_s+loss_D_t
        sum_loss.backward()
        optimizer.step()
        optimizer_im.step()
        optimizer_D.step()
        time_taken=time.time()
        time_elapsed=time_taken-time_start

        #print(
        #'iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv = {3:.4f}, loss_weight = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f}, bottle_neck_loss_s={7:4f},beta_s={8:4f},bottle_neck_loss_t={9:4f},beta_t={10:4f},ssim_mu={11:4f},ssim_sigmas={12:4f}'.format(
        #    i_iter, args.num_steps, loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t,bottle_neck_loss_s,beta_s,bottle_neck_loss_t,beta_t,ssim_mu,ssim_sigmas), 'Time elapsed ', time_elapsed)
        print(
        'iter = {0:6d}/{1:6d}, loss_seg = {2:.4f} loss_adv2 = {3:.4f}, loss_weight = {4:.4f}, loss_D_s = {5:.4f} loss_D_t = {6:.4f},style_loss1= {7:.4f},style_loss2={8:.4f},content_loss= {9:.4f},loss_adv1 = {10:.4f}'.format(
            i_iter, args.num_steps, loss_seg, loss_adv2, loss_weight, loss_D_s, loss_D_t,style_loss1,style_loss2,content_loss,loss_adv1), 'Time elapsed ', time_elapsed)

        f_loss = open(osp.join(args.snapshot_dir,'loss.txt'), 'a')
        f_loss.write('{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f}\n'.format(
            loss_seg, loss_adv, loss_weight, loss_D_s, loss_D_t))
        f_loss.close()
        
        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(args.num_steps) + '_D.pth'))
            torch.save(im1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_im.pth'))

            break
        if i_iter % 1000 == 0 and i_iter >20000:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))
            torch.save(im1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_im.pth'))
            torch.save({'Iter':i_iter,  'optimizer_state_dict':optimizer.state_dict(),  'optimizerD_state_dict':optimizer_D.state_dict(),  'optimizerim_state_dict':optimizer_im.state_dict()}, osp.join(args.snapshot_dir, 'GTA5_optimizers_final_cont.tar'))
            continue
        if i_iter % args.save_pred_every == 0 and i_iter != 0:
            print('taking snapshot ...')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '.pth'))
            torch.save(model_D.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_D.pth'))
            torch.save(im1.state_dict(), osp.join(args.snapshot_dir, 'GTA5_' + str(i_iter) + '_im.pth'))
            torch.save({'Iter':i_iter,  'optimizer_state_dict':optimizer.state_dict(),  'optimizerD_state_dict':optimizer_D.state_dict(),  'optimizerim_state_dict':optimizer_im.state_dict()}, osp.join(args.snapshot_dir, 'GTA5_optimizers_final_cont.tar'))
if __name__ == '__main__':
    main()

