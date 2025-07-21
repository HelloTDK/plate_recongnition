from ultralytics import YOLO
import cv2
import os

if __name__ == "__main__":
    model = YOLO("../weights/yolo_det.pt")
    image = cv2.imread("../data/imgs/1.jpg")
    results = model.predict(image)
    print(results)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()