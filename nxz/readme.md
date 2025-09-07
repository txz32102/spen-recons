## connect to server

```bash
ssh zjr@192.168.1.103
sudo -i
su hs3
cd /home/user/NewSpace/nxz/dwi/CycleGAN/MR-motion-bootstrap-subsampling/MR_motion_bootstrap_subsampling-main
```

## copy file

```bash
# musong to zjr
scp "/home/user/NewSpace/nxz/dwi/CycleGAN/MR-motion-bootstrap-subsampling/MR_motion_bootstrap_subsampling-main/Results_2/ZhenShi_rat_2_1_1/test_N/train_RealData_2.mat" \
musong@192.168.1.104:/home/data1/musong/workspace/python/spen-recons/nxz/copied_file/
# zjr to musong
```

## meaningful data

2025-9-7，我主要找了旧服务器(192.168.1.103)上在`/home/user/NewSpace/nxz/dwi/CycleGAN/MR-motion-bootstrap-subsampling/MR_motion_bootstrap_subsampling-main/our_data_2`的数据，`/home/user/NewSpace/nxz/dwi/CycleGAN/MR-motion-bootstrap-subsampling/MR_motion_bootstrap_subsampling-main/our_data_2/woPhase/real_data/test/test_SPENAZ_185.mat`可能是实采的小鼠数据，但是我似乎看到了论文里不同模态的对应图片？