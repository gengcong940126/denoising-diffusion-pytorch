from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os

def main(train):
    if train==True:
        model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8)
        ).cuda()

        diffusion = GaussianDiffusion(
            model,
            image_size = 64,
            timesteps = 1000,   # number of steps
            loss_type = 'l1'    # L1 or L2
        ).cuda()

        trainer = Trainer(
            diffusion,
            '/dtu-compute/congen/animeface',
            train=True,
            dataset_name='anemiface',
            image_size=64,
            train_batch_size = 64,
            train_lr = 2e-5,
            train_num_steps = 300001,         # total training steps
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            fp16 = True                   # turn on mixed precision training with apex
        )
        trainer.load(15)
        trainer.train()
    else:
        model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8)
        ).cuda()

        diffusion = GaussianDiffusion(
            model,
            image_size=64,
            timesteps=1000,  # number of steps
            loss_type='l1'  # L1 or L2
        ).cuda()

        trainer = Trainer(
            diffusion,
            '/dtu-compute/congen/animeface',
            train=False,
            dataset_name='anemiface',
            image_size=64,
            train_batch_size=100,
            train_lr=2e-5,
            train_num_steps=150001,  # total training steps
            gradient_accumulate_every=2,  # gradient accumulation steps
            ema_decay=0.995,  # exponential moving average decay
            fp16=True  # turn on mixed precision training with apex
        )
        trainer.test_anemiface()


"""
    Usage:

        export CUDA_VISIBLE_DEVICES=1
        export PORT=6006
        export CUDA_HOME=/opt/cuda/cuda-10.2
        export TIME_STR=1
        python train_anemiface.py 
        


    :return:
    """
if __name__ == '__main__':
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    main(False)