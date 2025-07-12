import cv2

def find_available_cameras(max_index=10):
    """
    检测可用摄像头索引。
    """
    available_cameras = []
    print("🔍 正在检测可用摄像头...\n")
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ 摄像头可用: ID = {i}")
            available_cameras.append(i)
            cap.release()
        else:
            print(f"❌ 摄像头不可用: ID = {i}")
    return available_cameras

def test_camera(camera_id=0):
    """
    打开指定摄像头并显示画面。
    """
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"\n❌ 无法打开摄像头（ID: {camera_id}）")
        return

    print(f"\n✅ 成功打开摄像头（ID: {camera_id}），按 'q' 键退出")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️ 无法读取摄像头帧")
                break

            cv2.imshow("Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 摄像头已释放，窗口已关闭")

def main():
    cameras = find_available_cameras()

    if not cameras:
        print("\n🚫 未找到任何可用摄像头。")
        return

    print(f"\n可用摄像头列表: {cameras}")
    try:
        camera_id = int(input("请输入要测试的摄像头 ID（默认 0）: ") or "0")
    except ValueError:
        print("⚠️ 输入无效，默认使用 ID = 0")
        camera_id = 0

    test_camera(camera_id)

if __name__ == "__main__":
    main()
