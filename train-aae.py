import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
import wandb
import matplotlib.pyplot as plt

import argparse
import random
from tqdm import tqdm

from models import weights_init, Discriminator, Generator
from operation import copy_G_params, load_params, get_dir
from operation import ImageFolder, InfiniteSamplerWrapper
from diffaug import DiffAugment
policy = 'color,translation'
import lpips
import time

from modified_models import DirectDiscriminator
from modified_models import TransposeDiscriminator

percept = None
_start_time = time.time()

import generative_model_score

def tic():
    global _start_time
    _start_time = time.time()

def tac(rounding=True):
    if rounding:
        t_sec = round(time.time() - _start_time)
    else:
        t_sec = time.time() - _start_time
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour,t_min,t_sec))

def crop_image_by_part(image, part):
    hw = image.shape[2]//2
    if part==0:
        return image[:,:,:hw,:hw]
    if part==1:
        return image[:,:,:hw,hw:]
    if part==2:
        return image[:,:,hw:,:hw]
    if part==3:
        return image[:,:,hw:,hw:]
    
import pytorch_msssim
def ssim_or_l1(img, real_image, alpha=0.84):
    if img.shape[2] < 32 : 
        img = F.interpolate(img, 32)
    interpor = F.interpolate(real_image, img.shape[2])
    sim = pytorch_msssim.msssim(img, interpor, normalize='relu')
    l1= torch.abs(img - interpor).mean()
    if sim != 0 :
        return -sim*alpha + l1*(1-alpha)
    else :
        return l1

def train_d(net, data, label="real"):
    """Train function of discriminator"""
    if label=="real":
        part = random.randint(0, 3)
        pred, [rec_all, rec_small, rec_part] = net(data, label, part=part)
        err = F.relu(  torch.rand_like(pred) * 0.2 + 0.8 -  pred).mean() + \
            percept( rec_all, F.interpolate(data, rec_all.shape[2]) ).sum() +\
            percept( rec_small, F.interpolate(data, rec_small.shape[2]) ).sum() +\
            percept( rec_part, F.interpolate(crop_image_by_part(data, part), rec_part.shape[2]) ).sum()
        err.backward()
        return pred.mean().item(), rec_all, rec_small, rec_part
    else:
        pred = net(data, label)
        err = F.relu( torch.rand_like(pred) * 0.2 + 0.8 + pred).mean()
        err.backward()
        return pred.mean().item()
        
import numpy as np        
to_img = lambda x : ((torch.clip(x, -1, 1)+1)/2*255).numpy().astype(np.uint8)

def train(args):

    data_root = args.path
    total_iterations = args.iter
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    ndf = 64
    ngf = 64
    nz = 256
    nlr = 0.0002
    nbeta1 = 0.5
    use_cuda = True
    multi_gpu = False
    dataloader_workers = 8
    current_iteration = 0
    save_interval = 10
    saved_model_folder, saved_image_folder = get_dir(args)
    
    device = torch.device(args.device)
    
    global percept
    percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True, gpu_ids=[device])
    
    transform_list = [
            transforms.Resize((int(im_size),int(im_size))),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
    trans = transforms.Compose(transform_list)
    
    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = ImageFolder(root=data_root, transform=trans)

    # cpu 성능이 기본 세팅에 비해 부족하여, num_workers와 pin_memory 옵션을 제거함
    # dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
    #                   sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers, pin_memory=True))
    #dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
    #                             sampler=InfiniteSamplerWrapper(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                            num_workers=16, pin_memory=True)
    
    score_model = generative_model_score.GenerativeModelScore(dataloader)
    score_model.lazy_mode(True) 
    score_model.load_or_gen_realimage_info(device)
    
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''
    
    
    #from model_s import Generator, Discriminator
    netG = Generator(ngf=ngf, nz=nz, im_size=im_size)
    netG.apply(weights_init)

    netD = Discriminator(ndf=ndf, im_size=im_size)
    netD.apply(weights_init)

    netG.to(device)
    netD.to(device)
    
    netFE = torch.nn.Sequential(
            torch.nn.Conv2d(3,3,4,2,1),
            torch.nn.Conv2d(3,3,4,2,1),
            torch.nn.Conv2d(3,1,4,2,1),
        ).to(device)
    netFD = DirectDiscriminator().to(device)
    netTFD = TransposeDiscriminator().to(device)
    netM = torch.nn.Linear(256,256).to(device)

    avg_param_G = copy_G_params(netG)

    fixed_noise = torch.FloatTensor(8, nz).normal_(0, 1).to(device)
    
    if checkpoint != 'None':
        ckpt = torch.load(checkpoint)
        netG.load_state_dict(ckpt['g'])
        netD.load_state_dict(ckpt['d'])
        avg_param_G = ckpt['g_ema']
        optimizerG.load_state_dict(ckpt['opt_g'])
        optimizerD.load_state_dict(ckpt['opt_d'])
        current_iteration = int(checkpoint.split('_')[-1].split('.')[0])
        del ckpt
        
    if multi_gpu:
        netG = nn.DataParallel(netG.to(device))
        netD = nn.DataParallel(netD.to(device))

    optimizerG = optim.Adam(netG.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerFE =  torch.optim.Adam(netFE.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    optimizerFD =  torch.optim.Adam(list(netFD.parameters()) + list(netTFD.parameters()), lr=nlr, betas=(nbeta1, 0.999))
    optimizerM =  torch.optim.Adam(netM.parameters(), lr=nlr, betas=(nbeta1, 0.999))
    
    if args.wandb : 
        wandb.init(project='FastGan', config=args)
    
    for iteration in tqdm(range(current_iteration, total_iterations+1)):
        real_image = next(iter(dataloader))
        real_image = real_image.to(device)
        real_image = DiffAugment(real_image, policy=policy)
        
        
        with torch.no_grad():
            mapped_feature = netM(torch.randn(data.size(0), 256, device=device))
        
        with torch.set_grad_enabled(True):
            # train Discriminator for image
            #fake_images = netG(mapped_feature), case='rec_all')
            fake_images = netG(mapped_feature)
            fake_images = [DiffAugment(fake, policy=policy) for fake in fake_images]
            
            err_dr, rec_img_all, rec_img_small, rec_img_part = train_d(netD, real_image, label="real")

            optimizerD.zero_grad()
            train_d(netD, [fi.detach() for fi in fake_images], label="fake")
            optimizerD.step()
        
            # train Generator whole part for Image generation
            netG.zero_grad()
            pred_g = netD(fake_images, "fake")
            err_g = -pred_g.mean()
            err_g.backward()
            gen_images = fake_images[0][0].detach().cpu()
        
        torch.cuda.empty_cache()    
        with torch.set_grad_enabled(True):
            # train Generator and Encoder for image reconsturction
            encoded_feature = netFE(F.interpolate(real_image, 128)).view(-1,256)
            fake_images = netG(encoded_feature)
            if epoch < 10 : 
                percept_loss = ssim_or_l1(fake_images[0], real_image)
            else : 
                percept_loss = percept(fake_images[0], real_image).mean()
     
    
        with torch.set_grad_enabled(True):
            # train Generator and Encoder for image reconsturction
            encoded_feature = netFE(F.interpolate(real_image, 128)).view(-1,256)
            fake_images = netG(encoded_feature)
            if epoch < 10 : 
                percept_loss = ssim_or_l1(fake_images[0], real_image)
            else : 
                percept_loss = percept(fake_images[0], real_image).mean()
            #ssim_loss = ssim_or_l1(fake_images[0], real_image)
            
            
            #err_E_G = torch.abs(rec_image - small_size_real_image).mean() +  torch.abs(fake_image - real_image).mean() #percept( fake_image, real_image ).sum() #- pred_g.mean()
            err_E_G = percept_loss #+ rec16_loss + (rec32_loss +rec64_loss + rec128_loss + rec256_loss)
            optimizerFE.zero_grad()
            optimizerG.zero_grad()
            err_E_G.backward()
            optimizerFE.step()
            optimizerG.step()

            # train FeatureDiscriminator for feature
            err_FD = (netFD(encoded_feature.detach())-1).mean()**2 + netFD(mapped_feature.detach()).mean()**2 +\
                     (netTFD(encoded_feature.detach())-1).mean()**2 + netTFD(mapped_feature.detach()).mean()**2
            optimizerFD.zero_grad()
            err_FD.backward()
            optimizerFD.step()

            # train Mapper
            mapped_feature = netM(torch.randn(data.size(0), 256, device=device))
            err_M = (netFD(mapped_feature)-1).mean()**2  + (netTFD(mapped_feature)-1).mean()**2
            optimizerM.zero_grad()
            err_M.backward()
            optimizerM.step()
    
        for p, avg_p in zip(netG.parameters(), avg_param_G):
            avg_p.mul_(0.999).add_(0.001 * p.data)

        if iteration % 100 == 0:
            print("GAN: loss d: %.5f    loss g: %.5f"%(err_dr, -err_g.item()))

        if iteration % (save_interval*10) == 0:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            with torch.no_grad():
                vutils.save_image(netG(netM(fixed_noise))[0].add(1).mul(0.5), saved_image_folder+'/%d.jpg'%iteration, nrow=4)
                vutils.save_image( torch.cat([
                        F.interpolate(real_image, 128), 
                        rec_img_all, rec_img_small,
                        rec_img_part]).add(1).mul(0.5), saved_image_folder+'/rec_%d.jpg'%iteration )
            load_params(netG, backup_para)

        if iteration % (save_interval*50) == 0 or iteration == total_iterations:
            backup_para = copy_G_params(netG)
            load_params(netG, avg_param_G)
            torch.save({'g':netG.state_dict(),'d':netD.state_dict()}, saved_model_folder+'/%d.pth'%iteration)
            load_params(netG, backup_para)
            torch.save({'g':netG.state_dict(),
                        'd':netD.state_dict(),
                        'g_ema': avg_param_G,
                        'opt_g': optimizerG.state_dict(),
                        'opt_d': optimizerD.state_dict()}, saved_model_folder+'/all_%d.pth'%iteration)
            
            
            fake_images_list = []
            netG.eval()
            netM.eval()

            for data in tqdm(dataloader, desc="[Generative Score]gen fake image...") :
                with torch.no_grad() : 
                    noise = torch.Tensor(current_batch_size, nz).normal_(0, 1).to(device)
                    fake_images = netG(netM(noise))
                    fake_images_list.append(fake_images[0].cpu())

            netG.to('cpu')
            netD.to('cpu') 
            #loss_fn.to('cpu')

            fake_image_tensor = torch.cat(fake_images_list)
            score_model.model_to(device)
            score_model.lazy_forward(fake_forward=True, fake_image_tensor=fake_image_tensor, device=device)
            score_model.model_to('cpu')
            netG.to(device)
            netD.to(device)
            #loss_fn.to(device)
            score_model.calculate_fake_image_statistics()
            score_model.calculate_fake_image_statistics()
            metrics = score_model.calculate_generative_score()
            
            with torch.no_grad() : 
                fig, ax = plt.subplots(1,1, figsize=(8,32))
                ax.imshow(to_img(fake_images[0][0].detach().cpu().permute(1,2,0)))

            log_dict = {'err_dr' : err_dr,
                       'err_g' : err_g,
                       # 'sum_per_loss':sum_per_loss,
                       'fig' :  wandb.Image(fig),
                        #'density_fig' : wandb.Image(density_fig),
                        #'lr' : scheduler._last_lr[0],
                        #'sum_dis_loss_for_dis' : sum_dis_loss_for_dis,
                        #'sum_dis_loss_for_en' : sum_dis_loss_for_en,
                        #'dis_for_gaussian' : dis_for_gaussian.mean(),
                        #'dis_for_encoded' : dis_for_encoded.mean(),
                     }
            log_dict.update(metrics)
            if args.wandb : 
                wandb.log(log_dict)
            else : 
                print(log_dict)
            plt.close()
            
            
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='region gan')

    parser.add_argument('--path', type=str, default='../lmdbs/art_landscape_1k', help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--device', type=str, default='cuda:0', help='device name')
    parser.add_argument('--name', type=str, default='test1', help='experiment name')
    parser.add_argument('--iter', type=int, default=50000, help='number of iterations')
    parser.add_argument('--start_iter', type=int, default=0, help='the iteration to start training')
    parser.add_argument('--batch_size', type=int, default=8, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--ckpt', type=str, default='None', help='checkpoint weight path if have one')
    parser.add_argument('--wandb', type=bool, default=False, help='log wandb')

    args = parser.parse_args()
    print(args)

    tic()
    train(args)
    tac()
