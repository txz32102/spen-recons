netG_lr2hr = Generator(opt.input_nc, opt.output_nc)
netD_hr = Discriminator(opt.input_nc)
netD_lr = Discriminator(opt.input_nc)

PM = physical_model()

if opt.cuda:
    netG_lr2hr.cuda()
    netD_hr.cuda()

netG_lr2hr.apply(weights_init_normal)
netD_hr.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(netG_lr2hr.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_hr = torch.optim.Adam(netD_hr.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_hr = torch.optim.lr_scheduler.LambdaLR(optimizer_D_hr, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
device = torch.device("cuda" if opt.cuda else "cpu")
input_hr = torch.empty(opt.batchSize, opt.input_nc, opt.size, opt.size,
                       dtype=torch.float32, device=device)
input_lr = torch.empty(opt.batchSize, opt.output_nc, opt.size, opt.size,
                       dtype=torch.float32, device=device)

target_real = torch.ones((opt.batchSize, 1), dtype=torch.float32, device=device, requires_grad=False)
target_fake = torch.zeros((opt.batchSize, 1), dtype=torch.float32, device=device, requires_grad=False)

fake_hr_buffer = ReplayBuffer()

dataloader = DataLoader(SpenDataset(opt.dataroot, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)

os.makedirs(opt.log_dir, exist_ok=True)
os.makedirs(f'{opt.log_dir}/train', exist_ok=True)
# Loss plot
logger = Logger(opt.n_epochs, len(dataloader), f'{opt.log_dir}/train')
###################################

###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input
        real_hr = batch['hr'].to(device)
        real_lr = batch['lr'].to(device)

        optimizer_G.zero_grad()

        # GAN loss
        with torch.no_grad():
            pm_lr_1 = PM((real_hr+1)/2)
        pm_lr_1 = _complex_to_1ch(pm_lr_1)
        fake_hr = netG_lr2hr(pm_lr_1)
        pred_fake = netD_hr(fake_hr)
        
        loss_GAN_hr2lr = criterion_GAN(pred_fake, target_real)

        # Cycle loss
        recovered_hr = netG_lr2hr(real_lr)
        with torch.no_grad():
            pm_lr_2 = PM((recovered_hr+1)/2)
        pm_lr_2 = _complex_to_1ch(pm_lr_2)
        loss_cycle_lrhrlr = criterion_cycle(pm_lr_2, real_lr)*5.0
        loss_cycle_hrlrhr = criterion_cycle(recovered_hr, real_hr)*5.0
        
        # Total loss
        loss_G = loss_GAN_hr2lr + loss_cycle_lrhrlr + loss_cycle_hrlrhr
        loss_G.backward()
        
        optimizer_G.step()
        ###################################

        ###### Discriminator hr ######
        optimizer_D_hr.zero_grad()

        # Real loss
        pred_real = netD_hr(real_hr)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_hr = fake_hr_buffer.push_and_pop(fake_hr)
        pred_fake = netD_hr(fake_hr.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D = (loss_D_real + loss_D_fake)*0.5
        loss_D.backward()

        optimizer_D_hr.step()
        
        if i == len(dataloader) - 1:
            logger.log({'loss_G': loss_G, 'loss_GAN_hr2lr': loss_GAN_hr2lr,
                        'loss_cycle_lrhrlr': loss_cycle_lrhrlr, 
                        'loss_cycle_hrlrhr': loss_cycle_hrlrhr,
                        'loss_D': loss_D}, 
                        images={'real_hr': real_hr, 
                                'real_lr': real_lr, 
                                'fake_hr': fake_hr, 
                                'fake_lr':pm_lr_2})
        else:
            logger.log({'loss_G': loss_G, 'loss_GAN_hr2lr': loss_GAN_hr2lr,
            'loss_cycle_lrhrlr': loss_cycle_lrhrlr, 
            'loss_cycle_hrlrhr': loss_cycle_hrlrhr,
            'loss_D': loss_D})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_hr.step()


    if (epoch % opt.ckpt_save_freq) == 0:
        os.makedirs(f'{opt.log_dir}/weights', exist_ok=True)

        torch.save(netG_lr2hr.state_dict(), f'{opt.log_dir}/weights/netG_lr2hr.pth')
        torch.save(netD_hr.state_dict(), f'{opt.log_dir}/weights/netD_hr.pth')

        print(f"[Checkpoint] Saved models at epoch {epoch}")

above code has some problems,
basically, i want to make a style conversion that can turn lr to hr and turn hr to lr

but instead of using two generators, i want to use only one generator since we has a physical
model that can turn hr to lr, the true lr, which is from the dataloader, is a little bit from the pm_lr_1 (which equals to PM((real_hr+1)/2)), but they are very close

the pipeline should be

hr->pm->pm_lr_1->generator->hr

lr(which is from the dataloader)->generator->hr->pm->lr

so eventually, we should has a generator, two discriminators

