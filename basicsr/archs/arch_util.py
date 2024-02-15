import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from torchvision.utils import save_image
from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


def make_layer_lp(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks with local padding.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return Sequential_LP(*layers)


class Sequential_LP(nn.Sequential):
    
    def forward(self, input,image_location ='None'):
        for module in self:
            input = module(input,image_location)
        return input


def crop_images(img, cropping_size_h=256, cropping_size_w=256, stride=256, device='cpu'):
    """
    Crop input images in a PyTorch tensor into smaller patches and concatenate them together.

    Args:
        img (torch.Tensor): Input tensor containing a batch of images with shape (N, C, H, W).
        cropping_size_h (int): Height of the cropped patches (default is 256).
        cropping_size_w (int): Width of the cropped patches (default is 256).
        stride (int): Stride for sliding the cropping window (default is 256).
        device (str): Device on which to perform the cropping and concatenation (default is 'cpu').

    Returns:
        torch.Tensor: Tensor containing concatenated patches with shape (N * P, C, cropping_size_h, cropping_size_w),
                     where P is the number of patches per image.
    """
    
    # Clone the input tensor to avoid modifying the original data
    img = img.clone()

    # Get the number of images in the batch
    N = img.shape[0]

    # Initialize an empty tensor to store the concatenated patches
    batch_patches = torch.tensor([]).to(device)

    # Loop through each image in the batch
    for l in range(N):
        # Crop the image into smaller patches
        crops = crop_image(img[l, :, :, :], cropping_size_h=cropping_size_h, cropping_size_w=cropping_size_w, stride=stride, device=device)

        # Concatenate the patches to the batch_patches tensor
        batch_patches = torch.cat((batch_patches, crops), 0)

    return batch_patches
   
def merge_patches_into_image(patches, num_rows=3, num_cols=3, device='cpu'):
    """
    Merge 2D patches into complete images.

    Args:
        patches (torch.Tensor): Input tensor of patches with shape (batch_size, channels, patch_height, patch_width).
        num_rows (int): Number of rows of patches in each image (default is 3).
        num_cols (int): Number of columns of patches in each image (default is 3).
        device (str): Device on which to perform the merging (default is 'cpu').

    Returns:
        torch.Tensor: Tensor containing merged images with shape (batch_size//num_patches_per_img, channels, height, width).
    """
    batch_size, channels, patch_height, patch_width = patches.size()
    num_patches_per_img = num_rows * num_cols

    # Calculate the dimensions of the merged images
    merged_height = patch_height * num_rows
    merged_width = patch_width * num_cols

    # Initialize an empty tensor to store the merged images
    merged_images = torch.empty((batch_size // num_patches_per_img, channels, merged_height, merged_width), device=device)

    for k in range(batch_size // num_patches_per_img):  # for each image
        merged_image = torch.tensor([]).to(device=device)

        for r in range(num_rows):  # each row in each image
            img_row = patches[k * num_patches_per_img + r * num_cols]

            for c in range(1, num_cols):
                img_row = torch.cat((img_row, patches[k * num_patches_per_img + r * num_cols + c]), dim=-1)

            merged_image = torch.cat((merged_image, img_row), dim=-2)  # concatenate rows

        merged_images[k] = merged_image

    return merged_images.to(device=device)
            


def crop_image(img, cropping_size_h=256, cropping_size_w=256, stride=256, device='cpu'):
    """
    Crop input image tensor into smaller patches.

    Args:
        img (torch.Tensor): Input tensor representing an image with shape (C, H, W).
        cropping_size_h (int): Height of the cropped patches (default is 256).
        cropping_size_w (int): Width of the cropped patches (default is 256).
        stride (int): Stride for sliding the cropping window (default is 256).
        device (str): Device on which to perform the cropping (default is 'cpu').

    Returns:
        torch.Tensor: Tensor containing cropped patches with shape (P, C, cropping_size_h, cropping_size_w),
                     where P is the number of patches.
    """
    # Get the height and width of the input image
    img_h = img.shape[1]
    img_w = img.shape[2]

    # Initialize an empty tensor to store the cropped patches
    crops = torch.tensor([]).to(device)

    # Initialize starting and ending indices for height
    start_h = 0
    end_h = cropping_size_h

    # Iterate over height with a sliding window
    while end_h <= img_h:
        # Initialize starting and ending indices for width
        start_w = 0
        end_w = cropping_size_w

        # Iterate over width with a sliding window
        while end_w <= img_w:
            # Crop the image
            crop = img[:, start_h:end_h, start_w:end_w].to(device)

            # Concatenate the crop to the crops tensor
            crops = torch.cat((crops, crop.unsqueeze(0)))

            # Update the width indices
            start_w += stride
            end_w += stride

        # Update the height indices
        start_h += stride
        end_h += stride

    return crops

class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


def tile_process(img,model,scale= 4,tile_size = 32,tile_pad = 8):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.

        Modified from: https://github.com/ata4/esrgan-launcher
        And then modified from https://github.com/xinntao/Real-ESRGAN/
        """
        batch, channel, height, width = img.shape
        output_height = height * scale
        output_width = width * scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = img.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)
        
        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                #print(input_tile.shape)
                # upscale tile
                try:
                    with torch.no_grad():
                        output_tile = model(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                
                # output tile area on total image
                output_start_x = input_start_x * scale
                output_end_x = input_end_x * scale
                output_start_y = input_start_y * scale
                output_end_y = input_end_y * scale

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * scale
                output_end_x_tile = output_start_x_tile + input_tile_width * scale
                output_start_y_tile = (input_start_y - input_start_y_pad) * scale
                output_end_y_tile = output_start_y_tile + input_tile_height * scale

                
                output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
                            
        return output



def super_resolve_from_gen_PatchByPatch_test(netG,input,num_patches_height=3, num_patches_width=3,generator_patch_resolution=8,SR_scale = 4, device ='cpu'): 
    
    """
    Generate a large image using a Patch-by-Patch sampling approach from a PyTorch generator network.

    This function super-resolves the input image in steps, where each step involves super resolving a sub-image.
    The outer patches in each sub_image is padded using outer_padding (zeros or replicate padding).
    These patches are re-super-resolved in the next sub-image with local padding and the previous outer patches are dropped.
    The output super-resolution sub-images are concatenated first vertically to form rows and then generation continues row by row.
    Finally, the rows are concatenated to form the complete large image.

    Parameters:
    - netG (nn.Module): The PyTorch generator network used for image generation.
    - num_patches_height (int): Number of patches along the height dimension (default is 3).
    - num_patches_width (int): Number of patches along the width dimension (default is 3).
    - device (str): Device on which to perform the generation (default is 'cpu').

    Returns:
    torch.Tensor: The super-resolved large image tensor of shape.
    """
    
    _,_,input_resolution_height,input_resolution_width = input.shape
   
    generator_output_patch_resolution = generator_patch_resolution*SR_scale
    output_resolution_height = input_resolution_height*SR_scale
    output_resolution_width = input_resolution_width*SR_scale
    
    sub_image_height = generator_patch_resolution*num_patches_height
    sub_image_width = generator_patch_resolution*num_patches_width
    
    # Calculate the number of steps in both dimensions required to iterate through the generator to generate the full image
    steps_h = math.ceil((input_resolution_height/generator_patch_resolution - 1)/(num_patches_height-1))
    steps_w = math.ceil((input_resolution_width/generator_patch_resolution - 1)/(num_patches_width-1))
    
    # Extend the input size by replicate padding to make sure it matches the target input defined by the calulated steps
    # In the end, we will drop those extended pixels
    extension_height = generator_patch_resolution*(steps_h*(num_patches_height-1)+1) - input_resolution_height
    extension_width = generator_patch_resolution*(steps_w*(num_patches_width-1)+1) - input_resolution_width 
    extended_input = F.pad(input, (0, extension_width, 0, extension_height), mode='replicate')

    # Build the inputs to the generator z and maps
    input_sub_images = crop_images(extended_input,sub_image_height,sub_image_width,generator_patch_resolution*(num_patches_width-1),'cpu')
    
    # Iterate through the generator with to generate the sub_images in sequence 
    # and concatente the sub_images to form the full image
    
    last_row_ind = steps_h-1
    last_column_ind = steps_w-1

    sub_image_ind = 0
    for ind_h in range(steps_h):
        for ind_w in range(steps_w):
            
            # Update the location of the generated image. Note that the first row could also be the last column
            if last_row_ind ==0:
                image_location = '1st_row_last_row'
            elif ind_h == 0:
                image_location = '1st_row'
            elif ind_h == last_row_ind :
                image_location = 'last_row'
            else:
                image_location = 'inter_row'
            
            if last_column_ind ==0:
                image_location += '_1st_col_last_col'
            elif ind_w ==0:
                image_location += '_1st_col'
            elif ind_w == last_column_ind :
                image_location += '_last_col'
            else:
                image_location += '_inter_col'
            
            # Get input for the current sub_image and crop it into patches
            input_sub_image = input_sub_images[[sub_image_ind]].to(device)
                        
            # Pass the input to the model to get patch_i
            with torch.no_grad():
                sub_image_i = netG(input_sub_image,image_location).cpu()

            # Drop the re-generated patches
            # Crop the left and bottom patches in the sub_image if it is not in the last row or last column
            if ind_h != last_row_ind and ind_w !=last_column_ind:
                sub_image_i_cropped = sub_image_i[:,:,0:generator_output_patch_resolution*(num_patches_height-1),0:generator_output_patch_resolution*(num_patches_width-1)]
                
            # Crop only the bottom patches in the sub_image if it is in the last column and not in the last row
            elif ind_h != last_row_ind:
                sub_image_i_cropped = sub_image_i[:,:,0:generator_output_patch_resolution*(num_patches_height-1),:]
                
            # Crop only the left patches in the sub_image if it is in the last row and not in the last column
            elif ind_w !=last_column_ind:
                sub_image_i_cropped = sub_image_i[:,:,:,0:generator_output_patch_resolution*(num_patches_width-1)]
                
            # Otherwise do not crop the image
            else:
                sub_image_i_cropped = sub_image_i
                    
            # Update the index for the next sub_image
            sub_image_ind = sub_image_ind+1
                        
            # Concatenate the sub images together to form a row
            if '1st_col' in image_location:
                image_row = sub_image_i_cropped
            else:
                image_row = torch.cat((image_row,sub_image_i_cropped),-1)
                
        # Concatenate the rows together to form the full image
        if '1st_row' in image_location:
            full_image = image_row
        else:
            full_image = torch.cat((full_image,image_row),-2)
            
    # Adjust the generated image to match the target size if it is larger
    full_image = full_image[:,:,:output_resolution_height,:output_resolution_width] 
    
    return full_image
    


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    ``Paper: Delving Deep into Deformable Alignment in Video Super-Resolution``
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
