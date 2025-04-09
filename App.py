from ultralytics import YOLO
import cv2

def detect_image(model, image_path):
    img = cv2.imread(image_path)
    results = model.predict(source=img, conf=0.5)
    annotated = results[0].plot()
    cv2.imshow("PPE Detection - Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    out = None

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter("output_video.avi", fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.5)
        annotated = results[0].plot()
        out.write(annotated)
        cv2.imshow("PPE Detection - Video", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def detect_webcam(model):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(source=frame, conf=0.5)
        annotated = results[0].plot()
        cv2.imshow("PPE Detection - Webcam", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

# MAIN
if __name__ == "__main__":
    model = YOLO("Yolo Weights/best.pt")

    print("\nChoose Mode:")
    print("1 - Detect on Image")
    print("2 - Detect on Video")
    print("3 - Real-time Detection via Webcam")
    choice = input("Enter choice (1/2/3): ")

    if choice == "1":
        image_path = input("Enter image path: ")
        detect_image(model, image_path)
    elif choice == "2":
        video_path = input("Enter video path: ")
        detect_video(model, video_path)
    elif choice == "3":
        detect_webcam(model)
    else:
        print("Invalid choice!")
