from imageai.Detection import ObjectDetection
import os


execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "yolo.h5"))
detector.loadModel()

custom_objects = detector.CustomObjects(bicycle=True, horse=True, boat=True)

detections = detector.detectCustomObjectsFromImage(custom_objects=custom_objects, input_image=os.path.join(execution_path , "image.jpg"), output_image_path=os.path.join(execution_path , "image1custom.jpg"), minimum_percentage_probability=1)

for eachObject in detections:
	print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
	print("--------------------------------")



