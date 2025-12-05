from google.cloud import vision
from google.api_core.client_options import ClientOptions

client = vision.ImageAnnotatorClient(client_options=ClientOptions(api_key='AIzaSyDSbaOh1Ujh9YzltDoGoU7bkjjvh3d8yQw'))

with open('example.jpg', 'rb') as f:
    content = f.read()

image = vision.Image(content=content)
response = client.label_detection(image=image)
labels = response.label_annotations

for label in labels:
    print(label.description, label.score)
