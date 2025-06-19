from inference import get_model
import cv2 as cv
import supervision as sv
import numpy as np

# Import the YOLO Model for Chess Piece Detection
model = get_model(model_id="chess.comdetection/4", api_key="API_KEY")

# Image of the chess Board
image = cv.imread(r"C:\Users\adity\OneDrive\Pictures\Screenshots\Screenshot 2025-06-15 174330.png")

# Getting the results of piece detection
results = model.infer(image=image)

# Prepare Annotators
detections = sv.Detections.from_inference(results[0])

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = bounding_box_annotator.annotate(scene=image.copy(),detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image,detections=detections)


# Display Result
sv.plot_image(annotated_image)
cv.waitKey(0)
cv.destroyAllWindows()
