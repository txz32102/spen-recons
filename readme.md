## useful training script
`/home/data1/musong/workspace/2025/8/08-20/tr/scripts/abs_lr_hr.py` is trainable


## bash training

```bash
cd /home/data1/musong/workspace/2025/8/08-20/tr
source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataroot /home/data1/musong/workspace/2025/8/08-20/tr/data/IXI_sim \
  --cuda \
  --batch_size 4 \
  --n_epochs 200 \
  --decay_epoch 100 \
  --n_blocks 2 \
  --base_ch 64 \
  --lr 2e-4 \
  --l_phase 1.0 \
  --l_img 1.0 \
  --l_dc 10.0 \
  --use_gan \
  --l_gan_phi 0.2 \
  --l_gan_img 0.5 \
  --outdir train

```


## test

```bash
cd /home/data1/musong/workspace/2025/8/08-20/tr
source /home/data1/anaconda3/bin/activate
conda activate /home/data1/musong/envs/main
CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataroot /home/data1/musong/workspace/2025/8/08-20/tr/test \
  --outdir test/test_result \
  --cuda
```