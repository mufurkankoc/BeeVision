import torch
import numpy as np
import cv2
from ultralytics import YOLO
from supervision import ColorPalette
from supervision import Detections, BoxAnnotator
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
    help="imaj kabul eden path")
args = vars(ap.parse_args())



class ObjectDetection:

    def __init__(self, capture_index):
       
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        
        self.model = self.load_model()
        
        self.CLASS_NAMES_DICT = self.model.model.names
    
        self.box_annotator = BoxAnnotator(thickness=1, text_thickness=1, text_scale=1)
    

    def load_model(self):
       
        model = YOLO("best.pt")  # pretrained YOLO yükleme
        model.fuse()
    
        return model


    def predict(self, frame):
       
        results = self.model(frame) # tahmi sonuçları
    
        return results
    
    def select_area(self, results,frame): # 1 etiket tahmini var ise ROI bölgesini return eder
        index=0                           # 1'den fazla tahmin için ROI alan bilgilerini return eder
        multiple_list=[]

        detection_number=self.count_detections(results)
        # print(detection_number)
        
        if detection_number == 1:
            xyxy=results[0][0].boxes.xyxy.cpu().numpy()
            area = list(xyxy)
            x1=int(area[0][0])
            x2=int(area[0][1])
            y1=int(area[0][2])
            y2=int(area[0][3])

            cropped_area = frame[x1:y2, x2:y1]        

            
       
        else:
            index=1
            for a,i in enumerate (results[0]):

                xyxy=results[0][a].boxes.xyxy.cpu().numpy()
                area = list(xyxy)
                x1=int(area[0][0])
                y1=int(area[0][1])
                x2=int(area[0][2])
                y2=int(area[0][3])
                
                liste=[x1,y1,x2,y2]
                multiple_list.append(liste)

                
        if index==0:
            return index, cropped_area
        else:
            return index, multiple_list


    def save_sticker(self, area): # ROI bölgesini ayrı bir imaj olarak kaydeder
        	
        grayImage = cv2.cvtColor(area, cv2.COLOR_BGR2GRAY)
        # dilated = cv2.erode(grayImage.copy(), None, iterations=1)
        # eroded = cv2.dilate(dilated, None, iterations=1)
        cv2.imwrite('ROI.png', grayImage)
    
    def count_detections(self,results):
        detect_number = len(results[0].boxes.xyxy.cpu().numpy())

        return detect_number
        


    def plot_bboxes(self, results, frame): # tahmin sonuçları için bbox çizer
        


        # Setup detections for visualization
        detections = Detections(
                    xyxy=results[0].boxes.xyxy.cpu().numpy(),
                    confidence=results[0].boxes.conf.cpu().numpy(),
                    class_id=results[0].boxes.cls.cpu().numpy().astype(int),
                    )
        
    
        # Format custom labels
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, tracker_id
        in detections]
        
        # Annotate and display frame
        frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels)
        
        return frame
    
    
    
    def __call__(self):

        if not args.get("image", False):
            args["image"] = "image 161012_1336.jpeg"
            print( "imaj yuklenemedi, program kapatılıyor." )
            # exit()

        frame = cv2.imread(args["image"])
        frame = cv2.resize(frame, (640, 640)) 
        results = self.predict(frame)
        frame = self.plot_bboxes(results, frame)
        cv2.imshow('YOLOv8 Detection', frame)

  
        size,cropped_area = self.select_area(results,frame)
        if size == 0:
            cv2.imshow('Cropped area', cropped_area)
            self.save_sticker(cropped_area)
            
        else:
            print("Tespit edilen etiket alanları")
            for i in cropped_area:
                print(i)

                
              
        cv2.waitKey(0)
   
        
        
    
detector = ObjectDetection(capture_index=0)
detector()
