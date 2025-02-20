## Multi-perspective Frequency Domain Learning for Deepfake Detection

## There is MFDL original code

---

### How to Train and Test

```bash
# How to Train
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot /root/yourpath/4_class_GAN_traindata --classes car,cat,chair,horse --batch_size 32 --delr_freq 10 --lr 0.001 --niter 85
```bash
# How to Test
CUDA_VISIBLE_DEVICES=0 python test.py --model_path ./MFDL.pth --batch_size 32
