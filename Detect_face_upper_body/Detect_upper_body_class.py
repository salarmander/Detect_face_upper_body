from ultralytics import YOLO

import os
import cv2

class PersonDetect:
    # Load model
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        self.model = YOLO(self.model_name)

    # Detect upper body
    def detect(self, video_path):

        if self.model is None:
            raise ValueError("Model is not loaded!")
        
        # Save direction for images of each video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        save_dir = os.path.join("Detect_face_upper_body/Data/Detect_output")
        output_dir = os.path.join(save_dir, video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        merge_dir = os.path.join("Detect_face_upper_body/Data/Merge_output")
        output_merge = os.path.join(merge_dir, video_name)
        os.makedirs(output_merge, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        saved = 0

        while True:
            # Read frame from video
            ret, frame = cap.read()
            if not ret:
                break

            # Use YOLO model to detect for each frame
            results = self.model(frame, verbose=False)

            #Export to image with boudingbox's upper body
            for box in results[0].boxes: # YOLO detect a frame for each times
                cls_id = int(box.cls[0]) # Take class id of object which is detected by YOLO
                if cls_id == 0:          # YOLO use COCO dataset, class 0 = "person"

                    # Cut video into full body image
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w = x2 - x1 # Height of image
                    h = y2 - y1 # Width of image
                    cropped = frame[y1:y2, x1:x2]

                    if cropped.size > 0:
                        # Draw boudingbox for upper body in each image
                        upper_body = int(h * 0.6)  # 60% of boudingbox
                        cv2.rectangle(cropped, (0, 0), (w, upper_body), (255, 0, 0), 2)

                        # Save export image after cut and set boudingbox
                        save_path = os.path.join(output_dir, f"img_{saved:04d}.jpg")
                        cv2.imwrite(save_path, cropped)
                        saved += 1

        cap.release()
        print(f"---------- Finish!!! Saved {saved} images to {output_dir}--------------")

