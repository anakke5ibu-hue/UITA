import cv2

def find_cameras(max_index=10):
    for i in range(max_index):
        # Gunakan backend DirectShow untuk Windows
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera found at index {i} and is accessible!")
            else:
                print(f"⚠️ Camera found at index {i}, but failed to grab frame.")
            cap.release()
        else:
            print(f"❌ No camera at index {i}")

if __name__ == "__main__":
    find_cameras(10)
    