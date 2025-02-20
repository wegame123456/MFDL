
import os
import torch
from util import Logger, printSet
from validate import validate
from networks.MFDL import mfdl
from options.test_options import TestOptions
import numpy as np

torch.manual_seed(42)




DetectionTests = {
                'Diffusion': { 'dataroot'   : '/root/yours/DM_testdata',
                                 'no_resize'  : False, 
                                 'no_crop'    : True,
                               },

           'GAN': { 'dataroot'   : '/root/yours/GAN_testdata',
                                 'no_resize'  : False,
                                 'no_crop'    : True,
                               },

                 }


opt = TestOptions().parse(print_options=False)
print(f'Model_path {opt.model_path}')

# get model
model = mfdl(num_classes=1)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")


model.load_state_dict(torch.load(opt.model_path, map_location='cpu'), strict=True)
model.cuda()
model.eval()


for testSet in DetectionTests.keys():
    dataroot = DetectionTests[testSet]['dataroot']
    # printSet(testSet)

    accs = [];aps = []
    for v_id, val in enumerate(os.listdir(dataroot)):
        opt.dataroot = '{}/{}'.format(dataroot, val)
        opt.classes  = '' 
        opt.no_resize = DetectionTests[testSet]['no_resize']
        opt.no_crop   = DetectionTests[testSet]['no_crop']
        acc, ap, _, _, _, _ = validate(model, opt)

        accs.append(acc);aps.append(ap)
        print("({} {:12}) acc: {:.1f}; ap: {:.1f}".format(v_id, val, acc*100, ap*100))
    print("({} {:10}) acc: {:.1f}; ap: {:.1f}".format(v_id+1,'Mean', np.array(accs).mean()*100, np.array(aps).mean()*100));print('*'*25) 

