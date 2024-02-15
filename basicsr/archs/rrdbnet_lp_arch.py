import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer_lp, pixel_unshuffle,crop_images,merge_patches_into_image

        
        
class conv2d_lp(nn.Module):
    """2D Conv supports local padding.

    Used in RRDB block in ESRGAN with local padding.

    Args:
        ch_in (int): number of input channels.
        ch_out (int): number of output channels
        padding_mode (str): padding mode used in the convolution either zeros or local
        merge_patches_into_image (bool): merge patches in to image if True when doing the local padding (if the input is not merged)
    """
    def __init__(self, ch_in, ch_out,padding_mode = 'zeros',merge_patches_into_image=True):
        super(conv2d_lp, self).__init__()
        self.padding_mode = padding_mode
        if padding_mode == 'local':
            self.local_padder = LocalPadder(merge_patches_into_image)
            self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 0)
        else:
            self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1)

    def forward(self, x, image_location= '1st_row_1st_col'):
        if self.padding_mode == 'local':
            x= self.local_padder(x,image_location)
        x = self.conv(x)
        
        return x

class LocalPadder(nn.Module):
    """
        PyTorch implementation of a Local Padder, which performs local padding based on convolutional settings.
        
        First the module merges the small-size patches together, perform outer padding and finally crop the patches with
        the specified overlapping padding size.

        Args:
            num_patches_h (int): Number of patches along the height dimension (default is 3).
            num_patches_w (int): Number of patches along the width dimension (default is 3).
            outer_padding (str): Padding mode for outer patches (default is 'constant').
            padding_size (int): Padding size for each patch (default is 1).
            conv_reduction (int): Reduction factor in spatial size after convolution (default is 2 for 3x3 conv).
        """
    num_patches_h = 3
    num_patches_w = 3
    outer_padding = 'constant'
    padding_size = 1
    conv_reduction = 2
        
    @classmethod
    def set_attributes(cls,num_patches_h = 3,num_patches_w =3,outer_padding = 'constant',padding_size =1,conv_reduction = 2):
        cls.num_patches_h = num_patches_h
        cls.num_patches_w = num_patches_w
        cls.outer_padding = outer_padding
        cls.padding_size = padding_size
        cls.conv_reduction = conv_reduction

    def __init__(self,merge_patches_into_image = True):
        super(LocalPadder, self).__init__()
        
        self.merge_patches_into_image = merge_patches_into_image
        # Initialize the padding variables to None
        self.vertical_padding_variable = None
        self.horizontal_padding_variable = None
        
        self.vertical_padding_variable_next_image = None
        self.horizontal_padding_variable_for_current_row = None
        self.horizontal_padding_variable_for_next_row= None
        
    def padding(self,input,image_location):
        
        #  Perform simple padding without padding variables during training or if they are the first patches to be generated during inference  
        if self.training or ('1st_row' in image_location and '1st_col' in image_location):
            output = F.pad(input, (self.padding_size,self.padding_size,self.padding_size,self.padding_size), self.outer_padding) # (_,_,3H+2,3W+2)

        # Pad only from left vertically if thery are the intermediate patches to be generated in the first row
        elif '1st_row' in image_location:
            output = torch.cat((self.vertical_padding_variable,input),-1) # (_,_,3H,3W+1)
            output = F.pad(output, (0,self.padding_size,self.padding_size,self.padding_size), self.outer_padding) # (_,_,3H+2,3W+2)
            # self.horizontal_padding_variable 

        # Pad only from top horizontally if thery are the first patches to be generated in subsequent rows (2nd, 3rd, ..)
        elif '1st_col' in image_location:
            output = F.pad(input, (self.padding_size,self.padding_size,0,self.padding_size), self.outer_padding) # (_,_,3H+1,3W+2)
            output = torch.cat((self.horizontal_padding_variable,output),-2) # (_,_,3H+2,3W+2)

        # Pad from left and top if thery are the intermediate patches to be generated in subsequent rows (2nd, 3rd, ..)
        else:
            output = torch.cat((self.vertical_padding_variable,input),-1) # (_,_,3H,3W+1)
            output = F.pad(output, (0,self.padding_size,0,self.padding_size), self.outer_padding) # (_,_,3H+1,3W+2)
            output = torch.cat((self.horizontal_padding_variable,output),-2) #  (_,_,3H+2,3W+2)
        
        return output
    
    def update_padding_variables(self,input,image_location,H,W):
        
        if self.vertical_padding_variable_next_image is not None:
            self.vertical_padding_variable = self.vertical_padding_variable_next_image
        
        # Get the vertical slice (_,_,3H,1) to be used as a vertical padding variable for the image in the next column
        # Discard the vertical padding variable if the image is in the last column
        if 'last_col' in image_location:
            self.vertical_padding_variable_next_image = None
        else:
            self.vertical_padding_variable_next_image = input[:,:,:,[W*(self.num_patches_w-1)-1]] 
        
        if 'last_col' in image_location:
            # (_,_,1,3W)
            horizontal_slice = input[:,:,[H*(self.num_patches_h-1)-1],:].cpu()
        else:
            # (_,_,1,2W)
            horizontal_slice =input[:,:,[H*(self.num_patches_h-1)-1],:W*(self.num_patches_w-1)].cpu()
        
        if '1st_col' in image_location:
            # For 2nd,3rd,.. rows, get the horizontal padding variable
            if '1st_row' not in image_location:
                self.horizontal_padding_variable_for_current_row = self.horizontal_padding_variable_for_next_row.clone()
                self.horizontal_padding_variable_for_current_row = F.pad(self.horizontal_padding_variable_for_current_row, (1,1,0,0), self.outer_padding)
            # Set the horizontal padding variable to the current horizontal_slice to be used for the next row
            
            self.horizontal_padding_variable_for_next_row = horizontal_slice
        else:
            # concatenate the horizontal slices from many passes to the model to form the horizontal_padding_variable for the next row
            self.horizontal_padding_variable_for_next_row = torch.cat((self.horizontal_padding_variable_for_next_row,horizontal_slice),-1)
    
                
        
        # Select the horizontal_padding_variable used for this image then
        # Update the current row variable for the next column or None if it is the last column
        if self.horizontal_padding_variable_for_current_row is not None:
            self.horizontal_padding_variable = self.horizontal_padding_variable_for_current_row[:,:,:,:self.num_patches_w*W+2].clone().to(input.device) # (_,_,1,3W+2)
            if 'last_col' in image_location:
                self.horizontal_padding_variable_for_current_row = None
            else:
                self.horizontal_padding_variable_for_current_row = self.horizontal_padding_variable_for_current_row[:,:,:,(self.num_patches_w-1)*W:]
                
    def forward(self, input,image_location='1st_row_1st_col'):
     
        _,_,H,W = input.size()
        # Merge patches into an image
        if self.merge_patches_into_image:
            merged_input = merge_patches_into_image(input,self.num_patches_h,self.num_patches_w,input.device) # (_,_,3W,3H)
        # If the input is merged then get the patch size
        else:
            H = H//self.num_patches_h
            W = W//self.num_patches_w
            merged_input = input

        # During inference only
        # Extract vertical and horizontal padding variables to be used for the next generation steps 
        if not self.training:
            self.update_padding_variables(merged_input,image_location,H,W)
        
        # Apply outer padding to the merged input as well as pad with stored padding variables from previous generation steps
        merged_input = self.padding(merged_input,image_location)

        # Perform cropping after padding to get the patches back, the cropping is done with an overlap to ensure local padding
        res_with_padding = W +self.padding_size*self.conv_reduction
        padded_output = crop_images(merged_input,res_with_padding,res_with_padding,W,device = input.device) # (_,_,H+2,W+2)
        

        return padded_output

class ResidualDenseBlock_lp(nn.Module):
    """Residual Dense Block with local_padding.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32,padding_mode='zeros'):
        super(ResidualDenseBlock_lp, self).__init__()
        
        self.padding = padding_mode         
        self.conv1 = conv2d_lp(num_feat, num_grow_ch,padding_mode)
        self.conv2 = conv2d_lp(num_feat + num_grow_ch, num_grow_ch,padding_mode)
        self.conv3 = conv2d_lp(num_feat + 2 * num_grow_ch, num_grow_ch,padding_mode)
        self.conv4 = conv2d_lp(num_feat + 3 * num_grow_ch, num_grow_ch,padding_mode)
        self.conv5 = conv2d_lp(num_feat + 4 * num_grow_ch, num_feat,padding_mode)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x,image_location = '1st_row_1st_col'):
        x1 = self.lrelu(self.conv1(x,image_location))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1),image_location))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1),image_location))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1),image_location))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1),image_location)
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x



class RRDB_lp(nn.Module):
    """Residual in Residual Dense Block with local padding.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32,padding_mode='zeros'):
        super(RRDB_lp, self).__init__()
        
        self.padding_mode = padding_mode
        
        self.rdb1 = ResidualDenseBlock_lp(num_feat, num_grow_ch,padding_mode)
        self.rdb2 = ResidualDenseBlock_lp(num_feat, num_grow_ch,padding_mode)
        self.rdb3 = ResidualDenseBlock_lp(num_feat, num_grow_ch,padding_mode)

    def forward(self, x,image_location='1st_row_1st_col'):
        out = self.rdb1(x,image_location)
        out = self.rdb2(out,image_location)
        out = self.rdb3(out,image_location)
        # Empirically, we use 0.2 to scale the residual for better performance
            
        return out * 0.2 + x
        


@ARCH_REGISTRY.register()
class RRDBNet_lp(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN that supports local padding.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
        TODO add local padding arguments
    """
    

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32,
                 padding_mode='zeros',outer_padding = 'constant',num_patches_h=3,num_patches_w=3):
        
        super(RRDBNet_lp, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
            
        self.padding_mode = padding_mode
        self.outer_padding = outer_padding
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        
        if padding_mode == 'local':
            LocalPadder.set_attributes(num_patches_h =num_patches_h ,num_patches_w=num_patches_w,outer_padding = outer_padding)

        #  As the input is passed as an image not patches, in the first convolution layer, do not merge the input. 
        self.conv_first = conv2d_lp(num_in_ch, num_feat,padding_mode,merge_patches_into_image = False)
        
        self.body = make_layer_lp(RRDB_lp, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch,padding_mode=padding_mode)
        self.conv_body = conv2d_lp(num_feat, num_feat,padding_mode)
        # upsample
        self.conv_up1 = conv2d_lp(num_feat, num_feat,padding_mode)
        self.conv_up2 = conv2d_lp(num_feat, num_feat,padding_mode)
        self.conv_hr = conv2d_lp(num_feat, num_feat,padding_mode)
        self.conv_last = conv2d_lp(num_feat, num_out_ch,padding_mode)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x,image_location = '1st_row_1st_col'):
        
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        
        feat = self.conv_first(feat,image_location)
        body_feat = self.body(feat,image_location)
        body_conv_feat = self.conv_body(body_feat,image_location)
        feat = feat + body_conv_feat
        
        # upsample
        feat_up = F.interpolate(feat, scale_factor=2, mode='nearest')
        feat_up = self.conv_up1(feat_up,image_location)
        feat_up = self.lrelu(feat_up)
        
        feat_up2 = F.interpolate(feat_up, scale_factor=2, mode='nearest')
        feat_up2 = self.conv_up2(feat_up2,image_location)
        feat_up2 = self.lrelu(feat_up2)
        
        feat_hr = self.conv_hr(feat_up2,image_location)
        feat_hr = self.lrelu(feat_hr)
        out = self.conv_last(feat_hr,image_location)
        
        out_merged = merge_patches_into_image(out,self.num_patches_h,self.num_patches_w,x.device)
        return out_merged
