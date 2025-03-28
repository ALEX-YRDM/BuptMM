import numpy as np
import cv2
import torch

def batch_nchw_to_bgr(tensor_batch):
    # Step 1: 确保Tensor在CPU上并转为NumPy
    np_batch = np.array(tensor_batch.detach().cpu().numpy())  # (N,C,H,W)
    
    # Step 2: 调整维度顺序 NCHW → NHWC
    nhwc_batch = np.transpose(np_batch, (0, 2, 3, 1))  # (N,H,W,C)
    
    # Step 3: 值域处理 (自动检测输入范围)
    if np.issubdtype(nhwc_batch.dtype, np.floating):
        nhwc_batch = (nhwc_batch * 255).clip(0, 255)
    uint8_batch = nhwc_batch.astype(np.uint8)
    
    # Step 4: 批量RGB → BGR转换
    bgr_images = []
    for img in uint8_batch:
        if img.shape[-1] == 3:  # 确保是3通道
            bgr_images.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        else:
            # 处理单通道情况 (复制为3通道)
            bgr_images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    
    return bgr_images  # 返回列表，每个元素是(H,W,3)

def bgr_to_nchw_tensor(bgr_array, normalize=True, device="cuda"):
    # Step 1: BGR → RGB 转换
    rgb_array = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2RGB)  # (H,W,3)

    # Step 2: 调整维度顺序 HWC → CHW
    chw_array = np.transpose(rgb_array, (2, 0, 1))  # (3,H,W)

    # Step 3: 值域处理
    if normalize:
        chw_array = chw_array.astype(np.float32) / 255.0  # [0,1]
    else:
        chw_array = chw_array.astype(np.float32)  # [0,255]

    # Step 4: 添加批次维度并转为Tensor
    tensor = torch.from_numpy(chw_array).unsqueeze(0)  # (1,3,H,W)
    
    return tensor.to(device)  # 转移到指定设备


def canny_process(output0,output1):
    # 转换为 CPU 并归一化到 0-255，再转 uint8
    output0 = np.array(batch_nchw_to_bgr(output0)[0])
    output1 = np.array(batch_nchw_to_bgr(output1)[0])
    edges_lq = cv2.Canny(output0, 280, 300)
    edges_mi = cv2.Canny(output1, 280, 300)
    enhance = cv2.bitwise_or(edges_mi, edges_lq)
    enhance = cv2.bitwise_xor(edges_mi, enhance)
    mask = (enhance == 255)
    mask = ~mask
    
    # 转换回 float32 进行计算
    output0 = output0.astype(np.float32)
    output1 = output1.astype(np.float32)
    output0[mask] = output1[mask].astype(np.float32) * 0.5 + output0[mask].astype(np.float32) * 0.5
    
    # 最终转换回 uint8和tansor
    return bgr_to_nchw_tensor(output0.astype(np.uint8))