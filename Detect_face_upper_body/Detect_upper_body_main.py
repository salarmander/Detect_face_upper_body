from Detect_upper_body_class import PersonDetect
import os

detector = PersonDetect("yolo11n-pose.pt")
detector.load_model()

video_folder = "Detect_face_upper_body/Data/Data_test/ShortVideo"
video_files = []
for f in os.listdir(video_folder):
    if f.lower().endswith((".mp4")):
        video_files.append(f)

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    print(f"Detecting: {video_path}")
    detector.detect(video_path)
