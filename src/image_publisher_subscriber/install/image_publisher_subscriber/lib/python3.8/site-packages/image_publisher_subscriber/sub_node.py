import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            'image_topic',  # Topic där bilder publiceras
            self.image_callback,
            10)
        self.subscription  # för att undvika varningar om oanvänd variabel

        # YOLOv8-modellen
        self.model = YOLO('yolov8n.pt')

        # CvBridge för att konvertera ROS2-bilder till OpenCV-format
        self.bridge = CvBridge()

    def image_callback(self, msg):
        # Konvertera ROS-bildmeddelandet till OpenCV-format
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Kör YOLOv8-analys på bilden
        results = self.model(frame)

        # Processera resultaten och skriv ut objekt och positioner
        self.process_results(results)

    def process_results(self, results):
        # Gå igenom varje resultat och objekt
        for result in results:
            for obj in result.boxes:
                # Hämta klassnamnet på objektet (t.ex. person, bil, etc.)
                label = self.model.names[int(obj.cls)]
                # Hämta bounding box-koordinater
                x1, y1, x2, y2 = map(int, obj.xyxy[0])
                # Skriv ut resultatet
                self.get_logger().info(f"Objekt: {label}, Position: [{x1}, {y1}, {x2}, {y2}]")

def main(args=None):
    rclpy.init(args=args)

    # Skapa noden och starta prenumerationen
    image_subscriber = ImageSubscriber()

    # ROS2-spin för att hålla noderna igång
    rclpy.spin(image_subscriber)

    # Stäng när noden är färdig
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
