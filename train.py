import torch
from image_loader import PairsLoader
from torch.utils.data import DataLoader
from reverse3d_prop import Reverse3dProp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from utils import phasemap_8bit


img_dir = '/home/wenbin/Documents/rgbd-supervision/phases_images_masks/Washington_scene_v2_1000samples'

image_res = (480, 640)
roi_res = (480, 640)
plane_idxs = [i for i in range(3)]
batch_size = 10


# Washington_Scene_V2_dataset = PairsLoader(data_path = img_dir, plane_idxs=plane_idxs, image_res=image_res, shuffle=True)
# train_loader = DataLoader(PairsLoader(os.path.join(img_dir, 'train'), plane_idxs=plane_idxs, image_res=image_res, shuffle=True))
# test_loader = DataLoader(PairsLoader(os.path.join(img_dir, 'test'), plane_idxs=plane_idxs, image_res=image_res, shuffle=True))


reverse_prop = Reverse3dProp()
reverse_prop = reverse_prop.cuda()
loss_fn = nn.MSELoss().cuda()

learning_rate = 1e-2
optimizer = torch.optim.SGD(reverse_prop.parameters(), lr=learning_rate)

epoch = 100000

total_train_step = 0
best_test_loss = float('inf')
best_test_psnr = 0

# 添加tensorboard
time_str = str(datetime.now()).replace(' ', '-').replace(':', '-')
# writer = SummaryWriter("./logs")
writer = SummaryWriter(f'runs/{time_str}')

writer.add_scalar("learning_rate", learning_rate)

for i in range(epoch):
    print(f"----------Training Start (Epoch: {i+1})-------------")
    
    # training steps
    reverse_prop.train()
    
    train_loader = DataLoader(PairsLoader(os.path.join(img_dir, 'train'), plane_idxs=plane_idxs, image_res=image_res, shuffle=True), batch_size=1)
    
    for img_mask_phase in train_loader:
        img, mask, phase = img_mask_phase
        img = img.cuda()
        mask = mask.cuda()
        phase = phase.cuda()
        
        masked_img = img*mask
        
        predicted_phase = reverse_prop(masked_img)    
        # predicted_phase = torch.tensor(phasemap_8bit(predicted_phase)).cuda()
        
        loss = loss_fn(predicted_phase, phase)
        
        # with torch.no_grad(): 
        #     psnr = utils.calculate_psnr(utils.target_planes_to_one_image(final_amp, masks), utils.target_planes_to_one_image(imgs, masks))
        
        writer.add_scalar("train_loss", loss.item(), total_train_step)
        # writer.add_scalar("train_psnr", psnr.item(), total_train_step)
        
        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step = total_train_step + 1
        if (total_train_step) % 1 == 0:
            
            print(f"Training Step {total_train_step}, Loss: {loss.item()}")
            
            # if (total_train_step) % 1000 == 0:
            #     writer.add_text('image id', imgs_id[0], total_train_step)
            #     for i in range(8):
            #         writer.add_image(f'input_images_plane{i}', imgs.squeeze()[i,:,:], total_train_step, dataformats='HW')
            #         writer.add_images(f'output_image_plane{i}', outputs_amp.squeeze()[i,:,:], total_train_step, dataformats='HW')
        
    # test the model after every epoch
    reverse_prop.eval()
    total_test_loss = 0
    test_items_count = 0
    with torch.no_grad():
        test_loader = DataLoader(PairsLoader(os.path.join(img_dir, 'test'), plane_idxs=plane_idxs, image_res=image_res, shuffle=True))
        for img_mask_phase in test_loader:
            img, mask, phase = img_mask_phase
            img = img.cuda()
            mask = mask.cuda()
            phase = phase.cuda()

            masked_img = img * mask
            
            predicted_phase = reverse_prop(masked_img)
            
            # outputs = reverse_prop(imgs)
            loss = loss_fn(predicted_phase, phase)
            
            total_test_loss += loss
            test_items_count += 1
        
        average_test_loss = total_test_loss/test_items_count
        if best_test_loss > average_test_loss:
            best_test_loss = average_test_loss
            # save model
            path = f"runs/{time_str}/model/"
            if not os.path.exists(path):
                os.makedirs(path) 
            torch.save(reverse_prop, f"runs/{time_str}/model/reverse_3d_prop_{time_str}_best_loss.pth")
            print("model saved!")
            
    print(f"Average Test Loss: {average_test_loss}")
    writer.add_scalar("average_test_loss", average_test_loss.item(), total_train_step)
    
    

# with torch.no_grad():
#     s = (final_amp * target_amp).mean() / \
#         (final_amp ** 2).mean()  # scale minimizing MSE btw recon and

# loss_val = loss_fn(s * final_amp, target_amp)