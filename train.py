from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os

def main(train):
    if train==True:
        model = Unet(
            dim = 128,
            dim_mults = (1, 2, 2, 2)
        ).cuda()

        diffusion = GaussianDiffusion(
            model,
            image_size = 32,
            timesteps = 1000,   # number of steps
            loss_type = 'l2'    # L1 or L2
        ).cuda()

        trainer = Trainer(
            diffusion,
            '/home/congen/code/geoml_gan/data/cifar10',
            train=True,
            dataset_name='cifar10',
            image_size=32,
            train_batch_size = 64,
            train_lr = 2e-4,
            train_num_steps = 500001,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            fp16 = True                   # turn on mixed precision training with apex
        )
        #trainer.load(20)
        trainer.train()
    else:

        model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8)
        ).cuda()

        diffusion = GaussianDiffusion(
            model,
            image_size=32,
            timesteps=1000,  # number of steps
            loss_type='l2'  # L1 or L2
        ).cuda()

        trainer = Trainer(
            diffusion,
            '/home/congen/code/geoml_gan/data/cifar10',
            train=False,
            dataset_name='cifar10',
            image_size=32,
            train_batch_size=64,
            train_lr=2e-4,
            train_num_steps=200001,  # total training steps
            gradient_accumulate_every=2,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            fp16=True  # turn on mixed precision training with apex
        )
        trainer.test()


"""
    Usage:

        export CUDA_VISIBLE_DEVICES=2
        export PORT=6006
        export CUDA_HOME=/opt/cuda/cuda-10.2
        export TIME_STR=1
        python train.py 


    :return:
    """
if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main(False)