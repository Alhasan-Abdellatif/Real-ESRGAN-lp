
from basicsr.archs.arch_util import super_resolve_from_gen_PatchByPatch_test,tile_process
from basicsr.archs.rrdbnet_lp_arch import RRDBNet_lp
from basicsr.archs.rrdbnet_arch import RRDBNet

import torch
import random
import os 
from PIL import Image

import cv2
import numpy as np

def test_single_image():
    LR_test_path = "../Data/srv/www/digrocks/portal/media/projects/215/origin/973/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_LR_default_X4"
    HR_test_path = "../Data/srv/www/digrocks/portal/media/projects/215/origin/973/images/DeepRockSR-2D/carbonate2D/carbonate2D_test_HR"


    # Get a list of file names in the specified folder
    file_names = os.listdir(HR_test_path)

    # Select a random image
    HR_image_name = random.choice(file_names)
    LR_image_name = HR_image_name[:-4]+'x4'+HR_image_name[-4:]
    # Create the full path to the selected image
    HR_image_path = os.path.join(HR_test_path, HR_image_name)
    LR_image_path = os.path.join(LR_test_path, LR_image_name)

    # Open and display the image using Pillow
    image_HR = Image.open(HR_image_path)
    image_LR = Image.open(LR_image_path)



    model_path = 'experiments/esrganLP_localpadding_bsgpu1_enlarge1/models/net_g_75000.pth'
    model_path = 'experiments/pretrained_models/ESRGAN/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth'
    model_path = 'experiments/esrganLP_local_padding_bsgpu1_enlarge1_gt256/models/net_g_75000.pth'


    device = 'cuda'

    device = torch.device(device)

    # set up model
    #model = RRDBNet(
    #    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

    #model = RRDBNet_lp(
    #    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)

    model = RRDBNet_lp(
        num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
        ,padding_mode= 'local',outer_padding= 'constant',num_patches_h= 4,num_patches_w= 4)

    model.load_state_dict(torch.load(model_path)['params'], strict=True)
    model.eval()
    model = model.to(device)
    
    
    output_folder = '../BasicSR/results/ESRGAN/'
    os.makedirs(output_folder, exist_ok=True)

    #for idx, path in enumerate(sorted(glob.glob(os.path.join(folder, '*')))):
    #    imgname = os.path.splitext(os.path.basename(path))[0]
    #    print(idx, imgname)

    #LR_image_path = '../../PhD/Codes/datasets/241.jpg'
    # read image
    img = cv2.imread(LR_image_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]],
                                        (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)

    #img = img[:,:,0:32,0:32]
    #img = img[:,:,0:32,0:56]
    #img = img[:,:,0:104,0:104]

    output = super_resolve_from_gen_PatchByPatch_test(model,img,4,4,16,device=device)
    #output = tile_process(img,model,4,32,8)
    #with torch.no_grad():
    #    output = model(img)
    print(output.shape)
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    output_name = f'{output_folder}output_ESRGAN2.png'
    cv2.imwrite(output_name, output)
    print('The image is saved as:', output_name)
    print('The HR image is :', HR_image_path)

    
    
if __name__ == '__main__':
    #root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_single_image() #(root_path)
