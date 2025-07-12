import cv2
import sys,os
import numpy as np
import matplotlib.pyplot as plt # 用于可视化，如果你的开发板环境支持

parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(parent_dir)

from src.models.lfnet import LFNet # 你的 OM 模型推理类


import pickle
from multiprocessing import shared_memory

# class
#     def __init__(self):
        
#     def pred
#         return

#     def show
# --- 预处理相关的全局参数 (与 lfnet.py 保持一致) ---
# 这些参数主要用于信息展示或外部预处理（如果 LFNet.infer 不自己做的话）
# 但由于 LFNet.infer 内部做了预处理，这里主要起文档作用
RESIZE_SHAPE = (512, 256) # (宽度, 高度)
# 确保 NORMALIZATION_MODE 与 lfnet.py 和训练时一致
NORMALIZATION_MODE = "minus_one_to_one"
# IMAGENET_MEAN 和 IMAGENET_STD 在 "minus_one_to_one" 模式下不需要

# ---------------------------------------------------------------------
# --- 从 PyTorch 测试脚本复制过来的后处理函数 ---
# ---------------------------------------------------------------------
def extract_lane_coordinates(binary_pred, instance_pred, min_pixels=50): # instance_pred 在这里可能没用到，但保持接口一致
    """提取车道线坐标函数"""
    # 注意：这里的 binary_pred 应该是从模型输出直接得到的，还未 resize 到原始图像大小
    # PyTorch 脚本中是在 resize 后的 binary_pred_resized 上做的，我们需要保持一致
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_pred.astype(np.uint8), connectivity=8
    )
    lane_coordinates = []
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_pixels:
            continue
        ys, xs = np.where(labels == label)
        lane_coordinates.append(list(zip(xs, ys))) # 存储的是在模型输出尺寸下的坐标
    return lane_coordinates

def fit_vertical_line(lane_points):
    """拟合垂直方向直线 x = ky + b（y为行坐标，x为列坐标）"""
    if not lane_points or len(lane_points) < 2: #至少需要两个点来拟合直线
        return None
    xs, ys = zip(*lane_points)
    ys = np.array(ys, dtype=np.float32)  # y坐标（行）
    xs = np.array(xs, dtype=np.float32)  # x坐标（列）

    # 构造拟合矩阵：xs = k * ys + b
    A = np.vstack([ys, np.ones(len(ys))]).T
    try:
        k, b = np.linalg.lstsq(A, xs, rcond=None)[0]
    except np.linalg.LinAlgError: # 如果矩阵奇异等问题
        return None
    return (k, b)  # 返回 (k, b)，对应 x = ky + b

def draw_vertical_line_on_mask_or_image(image_or_mask, line_params, color=(0, 0, 255), thickness=2): # 修改颜色以区分
    """根据垂直方向参数 (k, b) 在指定图像或掩码上绘制直线"""
    if line_params is None:
        return image_or_mask
    k, b = line_params
    height, width = image_or_mask.shape[:2]

    # 计算直线在图像顶部（y=0）和底部（y=height-1）的x坐标
    # 注意：这里的 y 对应的是 binary_pred_mask 的高度，即模型输出的高度
    y_top_mask = 0
    y_bottom_mask = height - 1 # mask 的高度

    x_top_mask = int(k * y_top_mask + b)
    x_bottom_mask = int(k * y_bottom_mask + b)

    # 直接在传入的 image_or_mask (可能是原始尺寸，也可能是模型输出尺寸) 上绘制
    # 如果是在原始图像上绘制，需要将 mask 坐标空间的直线参数转换
    # 但更简单的是，在与 binary_pred_mask相同尺寸的图像上绘制，然后一起resize
    # 或者，直接在resize后的图像上，使用原始图像尺寸的坐标空间来绘制
    # 为了与PyTorch脚本的visualize_results行为一致，我们将在最终的overlay图像上绘制
    # 这意味着line_params是基于原始图像尺寸的binary_pred_resized计算的
    # 因此，这里的height应该是原始图像的height

    # 修正：我们应该在原始图像尺寸上绘制直线
    # 因此，k,b 应该是基于 resize 到原始尺寸的 binary_pred_resized 计算的
    # 所以这里的 height 应该是原始图像的 height
    # x_top = int(k * 0 + b) # y=0 in original image
    # x_bottom = int(k * (original_image_height - 1) + b)
    # cv2.line(image_or_mask, (x_top, 0), (x_bottom, original_image_height - 1), color, thickness)

    # 为了简单起见，并与 PyTorch 脚本的 draw_vertical_line 保持一致
    # 我们假设这里的 image_or_mask 已经是目标绘制尺寸 (通常是原始图像尺寸)
    # 并且 line_params (k,b) 也是在该尺寸下计算得到的
    x_start = int(k * 0 + b)
    x_end = int(k * (height - 1) + b)
    cv2.line(image_or_mask, (x_start, 0), (x_end, height - 1), color, thickness)
    return image_or_mask
# ---------------------------------------------------------------------

def postprocess_om_outputs(om_outputs):
    """
    后处理 OM 模型输出。
    """
    if om_outputs is None or len(om_outputs) < 2:
        print("OM 模型输出格式不正确或为空。")
        return None, None

    binary_logits_np = om_outputs[0]
    instance_embedding_np = om_outputs[1]

    binary_pred_np = np.argmax(binary_logits_np, axis=1).squeeze(0)
    instance_pred_np = instance_embedding_np.squeeze(0) # (embedding_dim, H, W)

    return binary_pred_np, instance_pred_np # 返回的是模型输出尺寸的 mask

def visualize_segmentation_result(original_bgr_image, binary_pred_mask_model_size, alpha=0.5):
    """
    可视化二值分割结果。
    binary_pred_mask_model_size: 模型输出尺寸的二值掩码
    """
    binary_mask_uint8 = binary_pred_mask_model_size.astype(np.uint8)
    # 将模型输出尺寸的掩码 resize 到原始图像尺寸
    resized_binary_mask_for_viz = cv2.resize(
        binary_mask_uint8,
        (original_bgr_image.shape[1], original_bgr_image.shape[0]), # (width, height)
        interpolation=cv2.INTER_NEAREST
    )

    color_mask = np.zeros_like(original_bgr_image)
    color_mask[resized_binary_mask_for_viz == 1] = [0, 255, 0] # 绿色车道线

    overlay_image = cv2.addWeighted(original_bgr_image, 1, color_mask, alpha, 0)
    return overlay_image, resized_binary_mask_for_viz # 同时返回用于拟合的原始尺寸掩码

def display_image(window_name, image):
    """简单的图像显示函数，按任意键关闭"""
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def main():
    # 1. 加载 OM 模型
    om_model_path = './src/models/modelS_0.om' # 修改为你的OM模型路径
    try:
        model_handler = LFNet(om_model_path)
        print(f"成功加载 OM 模型: {om_model_path}")
    except Exception as e:
        print(f"加载 OM 模型失败: {e}")
        return

    # 2. 读取输入图像
    image_path = './src/models/1000.png' # 修改为你的测试图片路径
    original_bgr_image = cv2.imread(image_path)
    if original_bgr_image is None:
        print(f"错误: 无法读取图像文件 {image_path}")
        return

    original_height, original_width = original_bgr_image.shape[:2]
    print(f"原始图像形状: ({original_height}, {original_width})")
    # display_image('Original Image', original_bgr_image) # Matplotlib会显示，这里可以注释掉

    # 3. 模型推理
    try:
        # LFNet.infer 内部进行预处理，输入原始BGR图像
        om_outputs = model_handler.infer(original_bgr_image)
        print("OM 模型推理完成。")
    except Exception as e:
        print(f"OM 模型推理失败: {e}")
        return

    if om_outputs is None:
        print("模型推理没有返回结果。")
        return

    # 4. 后处理推理结果
    # binary_segmentation_mask 和 instance_segmentation_embedding 都是模型输出尺寸的
    binary_segmentation_mask_model_size, instance_segmentation_embedding_model_size = postprocess_om_outputs(om_outputs)

    if binary_segmentation_mask_model_size is None:
        print("后处理失败，无法获取分割掩码。")
        return

    print(f"二值分割掩码形状 (模型输出尺寸): {binary_segmentation_mask_model_size.shape}")
    if instance_segmentation_embedding_model_size is not None:
        print(f"实例分割嵌入形状 (模型输出尺寸): {instance_segmentation_embedding_model_size.shape}")

    # 5. 将模型输出的二值分割掩码 resize 到原始图像大小，用于提取坐标和可视化
    binary_pred_mask_original_size = cv2.resize(
        binary_segmentation_mask_model_size.astype(np.uint8),
        (original_width, original_height), # (width, height)
        interpolation=cv2.INTER_NEAREST # 对于掩码使用最近邻插值
    )
    print(f"二值分割掩码形状 (原始图像尺寸): {binary_pred_mask_original_size.shape}")


    # 6. 提取车道线坐标 (在原始图像尺寸的掩码上进行)
    # instance_pred 在这个函数中当前未使用，但为了匹配PyTorch脚本的接口，我们传入一个None或等效物
    # 如果你的 instance_segmentation_embedding 需要resize并且用于提取，你需要相应处理
    lanes_in_original_coords = extract_lane_coordinates(binary_pred_mask_original_size, None)

    # 7. 拟合垂直方向直线 (基于原始图像坐标系下的点)
    line_fits_params = []
    for lane_points in lanes_in_original_coords:
        line_params = fit_vertical_line(lane_points)
        
        if line_params is not None:
            line_fits_params.append(line_params)
            print(f"拟合直线参数 (k, b): {line_params}")
    print(f"结果列表：{line_fits_params}")

    


    # 8. 可视化分割结果和拟合的直线
    # 首先获取带分割掩码的叠加图像
    # 注意: visualize_segmentation_result 内部也会 resize 模型输出的掩码
    # 为了代码清晰，我们这里传入模型输出尺寸的掩码，让它内部resize
    result_image_overlay, _ = visualize_segmentation_result(original_bgr_image.copy(), binary_segmentation_mask_model_size)

    # 在叠加图像上绘制拟合的直线
    for params in line_fits_params:
        # draw_vertical_line_on_mask_or_image 期望的 image_or_mask 尺寸与 params (k,b) 的坐标系一致
        # params 是基于 original_size mask 计算的，所以应该在 original_size 的图像上绘制
        result_image_overlay = draw_vertical_line_on_mask_or_image(result_image_overlay, params, color=(255, 0, 0), thickness=2) # 用红色绘制拟合线

    print("结果可视化完成。")

    # 使用 Matplotlib 显示最终结果 (与 PyTorch 脚本一致)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(result_image_overlay, cv2.COLOR_BGR2RGB))
    plt.title('Lane Segmentation and Line Fitting (OM Model)')
    plt.axis('off')
    # 如果在没有GUI的环境（如某些开发板直接运行），plt.show() 可能无效或报错
    # 你可以取消注释下面的行来保存图片
    # plt.savefig("om_lane_detection_result.png")
    # print("结果已保存为 om_lane_detection_result.png")
    try:
        plt.show()
    except Exception as e:
        print(f"显示图像时出错 (可能是无 GUI 环境): {e}")
        output_filename = "om_lane_detection_result_with_lines.jpg"
        cv2.imwrite(output_filename, result_image_overlay)
        print(f"图像已保存到: {output_filename}")


    # cv2.destroyAllWindows() # 如果用了 plt.show(), 这个通常不需要



if __name__ == '__main__':
    main()