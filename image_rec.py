import os
from google.cloud import vision
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = '/content/industrial-joy-413218-81cc73ddb898.json'
client = vision.ImageAnnotatorClient()

image_path = '/content/CAMPUS.spring.6677-e1571237644789.jpg' 
with open(image_path, 'rb') as image_file:
    content = image_file.read()
image = vision.Image(content=content)

response = client.landmark_detection(image=image)

landmarks = response.landmark_annotations

if landmarks:
    for landmark in landmarks:
        print(f"Landmark detected: {landmark.description}")
        print(f"Confidence level: {landmark.score}")
        # Here you can add code to integrate the detected landmarks into your app
else:
    print("No landmarks detected or there was an error in the API call.")