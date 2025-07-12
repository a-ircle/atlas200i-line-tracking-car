import cv2
import sys,os
import numpy as np
import matplotlib.pyplot as plt # 用于可视化，如果你的开发板环境支持
import time

parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(parent_dir)

from src.models.lfnet import LFNet 

RESIZE_SHAPE = (512, 256) 
NORMALIZATION_MODE = "minus_one_to_one"

class LFModel:
    def __init__(self, path):
        try:
            self.model_handler = LFNet(path)
            print(f"成功加载 OM 模型: {path}")
        except Exception as e:
            print(f"加载 OM 模型失败: {e}")
            return
        
    def pred(self, img):
        start_t = time.time()
        om_outputs = self.model_handler.infer(img)
        end_t = time.time()
        print("time for only model is : ", end_t-start_t)
        # 后处理推理结果
        # binary_segmentation_mask 和 instance_segmentation_embedding 都是模型输出尺寸的
        binary_segmentation_mask_model_size, instance_segmentation_embedding_model_size = self.postprocess_om_outputs(om_outputs)
        
        original_height, original_width = img.shape[:2]
        binary_pred_mask_original_size = cv2.resize(
            binary_segmentation_mask_model_size.astype(np.uint8),
            (original_width, original_height), # (width, height)
            interpolation=cv2.INTER_NEAREST # 对于掩码使用最近邻插值
        )

        # 提取车道线坐标 (在原始图像尺寸的掩码上进行)
        # instance_pred 在这个函数中当前未使用，但为了匹配PyTorch脚本的接口，我们传入一个None或等效物
        # 如果你的 instance_segmentation_embedding 需要resize并且用于提取，你需要相应处理
        lanes_in_original_coords = self.extract_lane_coordinates(binary_pred_mask_original_size, None)

        # 拟合垂直方向直线 (基于原始图像坐标系下的点)
        line_fits_params = []
        top_two_lanes = self.f(lanes_in_original_coords)
        
        for lane_points in top_two_lanes:
            line_params = self.fit_vertical_line(lane_points)
            if line_params is not None:
                line_fits_params.append(line_params)
                print(f"拟合直线参数 (k, b): {line_params}")
        # print(f"结果列表：{line_fits_params}")
        
        result_image_overlay, _ = self.visualize_segmentation_result(img.copy(), binary_segmentation_mask_model_size)
        f_t = time.time()
        print("after pro : ", f_t - end_t)
      # 在叠加图像上绘制拟合的直线
        for params in line_fits_params:
          result_image_overlay = self.draw_vertical_line_on_mask_or_image(result_image_overlay, params, color=(255, 0, 0), thickness=2) # 用红色绘制拟合线
        
        cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)  # 允许窗口可调节大小
        cv2.resizeWindow("Camera Test", 1280, 720)        # 设置初始窗口大小（宽1280，高720）
        cv2.imshow('res', result_image_overlay)
        cv2.waitKey(1)
        
  
    #plt.imshow(cv2.cvtColor(result_image_overlay, cv2.COLOR_BGR2RGB))

    # 如果在没有GUI的环境（如某些开发板直接运行），plt.show() 可能无效或报错
    # 你可以取消注释下面的行来保存图片
    # plt.savefig("om_lane_detection_result.png")
    # print("结果已保存为 om_lane_detection_result.png")
        return line_fits_params

    def f(self, lanes_in_original_coords):
        lane_dis_list = []
        for lane_points in lanes_in_original_coords:
            if not lane_points or len(lane_points) < 2: #至少需要两个点来拟合直线
                continue
            xs, ys = zip(*lane_points)
            ys = np.array(ys, dtype=np.float32)  # y坐标（行）
            xs = np.array(xs, dtype=np.float32)  # x坐标（列）
            
            # 找到最小和最大 x 值的索引
            min_x_idx = np.argmin(xs)
            max_x_idx = np.argmax(xs)
    
            # 根据索引获取 x_min, x_max 及其对应的 y 值
            x_min = xs[min_x_idx]
            y_min = ys[min_x_idx]
    
            x_max = xs[max_x_idx]
            y_max = ys[max_x_idx]
    
            dis = (x_max - x_min)**2 +(y_max - y_min)**2
            lane_dis_list.append((dis, lane_points))

        # 根据 dis 从大到小排序，保留最大的两条线
        lane_dis_list.sort(reverse=True, key=lambda x: x[0])
        top_two_lanes = [item[1] for item in lane_dis_list[:4] if item]
    
        return top_two_lanes
            
    
    def postprocess_om_outputs(self, om_outputs):
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


    def extract_lane_coordinates(self, binary_pred, instance_pred, min_pixels=50): # instance_pred 在这里可能没用到，但保持接口一致
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


    def fit_vertical_line(self, lane_points):
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

# ---------------------------------------------------------------------
# --- 从 PyTorch 测试脚本复制过来的后处理函数 ---
# ---------------------------------------------------------------------



  
    def draw_vertical_line_on_mask_or_image(self, image_or_mask, line_params, color=(0, 0, 255), thickness=2): # 修改颜色以区分
        """根据垂直方向参数 (k, b) 在指定图像或掩码上绘制直线"""
        if line_params is None:
            return image_or_mask
        k, b = line_params
        height, width = image_or_mask.shape[:2]
    
        y_top_mask = 0
        y_bottom_mask = height - 1 # mask 的高度
    
        x_top_mask = int(k * y_top_mask + b)
        x_bottom_mask = int(k * y_bottom_mask + b)
    
        x_start = int(k * 0 + b)
        x_end = int(k * (height - 1) + b)
        cv2.line(image_or_mask, (x_start, 0), (x_end, height - 1), color, thickness)
        return image_or_mask
# ---------------------------------------------------------------------


  
    def visualize_segmentation_result(self, original_bgr_image, binary_pred_mask_model_size, alpha=0.5):
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