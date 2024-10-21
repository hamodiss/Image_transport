import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/root/Image_transport/src/yolo_image_subscriber/install/yolo_image_subscriber'
