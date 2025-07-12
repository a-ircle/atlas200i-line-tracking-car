import cv2,os

# 打开USB摄像头
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
try:
  cap = cv2.VideoCapture(0)
except:
  cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 初始化图像计数器
image_counter = 1000
save_dir = "/home/HwHiAiUser/E2E-Sample/Car/LLM/photo/Images/images15"
os.makedirs(save_dir, exist_ok=True)


print("按 'c' 键拍照，按 'q' 键退出程序。")

try:
    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        if not ret:
            print("无法从摄像头读取画面")
            break

        # 显示画面
        
	# 创建一个可调整大小的窗口
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

	# 调整窗口大小
        cv2.resizeWindow('Camera', 800, 600)  # 设定窗口大小为800x600像素

        cv2.imshow('Camera', frame)
        

        # 检测键盘输入
        key = cv2.waitKey(100) & 0xFF  # 等待1毫秒，并获取按键值
        # print(f"检测到的按键值: {key}")
        if key == ord('c'):
            # 保存图像
            img_name = f"/home/HwHiAiUser/E2E-Sample/Car/LLM/photo/Images/images15/{image_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"拍照成功，图片保存为 {img_name}")
            image_counter += 1

        elif key == ord('q'):
            # 按下 'q' 键退出
            print("退出程序")
            break

finally:
    # 释放摄像头并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()