import cv2
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
from PIL import Image

# Load the BLIP-2 model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

# Video file path
video_path = "input_video.mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

# Frame sampling rate (process every Nth frame)
frame_rate = 30  
frame_count = 0
captions = []

# Image preprocessing transformation
transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_rate == 0:
        # Convert frame to PIL image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)

        # Process image and generate caption
        inputs = processor(pil_image, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(**inputs)

        caption = processor.decode(output[0], skip_special_tokens=True)
        captions.append(f"Frame {frame_count}: {caption}")

        # Show the frame and caption (optional)
        cv2.putText(frame, caption, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save captions to a file
with open("video_captions.txt", "w") as f:
    for cap in captions:
        f.write(cap + "\n")

print("Video processing complete. Captions saved to video_captions.txt.")
