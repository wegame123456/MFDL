import os
import sys
import torch
import torch.nn
from data import create_dataloader
from MFDL.networks.MFDLtrainer import Trainer
from options.train_options import TrainOptions
from tqdm import tqdm





if __name__ == '__main__':
    opt = TrainOptions().parse()
    Testdataroot = os.path.join(opt.dataroot, 'test')
    opt.dataroot = '{}/{}/'.format(opt.dataroot, opt.train_split)
    print('  '.join(list(sys.argv)) )
    data_loader = create_dataloader(opt)

    
    model = Trainer(opt)
    model.train()
    for epoch in range(opt.niter):
        epoch_iter = 0
        with tqdm(total=len(data_loader), desc=f'Epoch {epoch + 1}/{opt.niter}', unit='batch') as pbar:
            for i, data in enumerate(data_loader):
                model.total_steps += 1
                epoch_iter += opt.batch_size

                model.set_input(data)
                model.optimize_parameters()
                pbar.update(1)

            if epoch % opt.delr_freq == 0 and epoch != 0:
                model.adjust_learning_rate()
                
        if (epoch+1) % 5 == 0: 
            torch.save(model.model.state_dict(),f'MFDL{epoch+1}.pth')

    
    
