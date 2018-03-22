import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import matplotlib.pyplot as plt

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2

import sys
sys.path.append('/home/arash/Desktop/models/research/object_detection')
sys.path.append('/home/arash/Desktop/models/research')
from utils import label_map_util
from utils import visualization_utils as vis_util

from AnimationBuilder import Player
#   ---------------------------------------
class VideoObjectDetection(object):

    def __init__(self, video_file, detection_graph):
        """
        Arguments:
            video_file: string, path to the video file
            detection_graph: tf graph object, tensorflow model

        VideoCapture object properties [indexed from 0 to 18:
            0: CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
            1: CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
            2: CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
            3: CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
            4: CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
            5: CAP_PROP_FPS Frame rate.
            6: CAP_PROP_FOURCC 4-character code of codec.
            7: CAP_PROP_FRAME_COUNT Number of frames in the video file.
            8: CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
            CAP_PROP_MODE Backend-specific value indicating the current capture mode.
            CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
            CAP_PROP_CONTRAST Contrast of the image (only for cameras).
            CAP_PROP_SATURATION Saturation of the image (only for cameras).
            CAP_PROP_HUE Hue of the image (only for cameras).
            CAP_PROP_GAIN Gain of the image (only for cameras).
            CAP_PROP_EXPOSURE Exposure (only for cameras).
            CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
            CAP_PROP_WHITE_BALANCE Currently not supported
            CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
        """

        self.video_file = video_file
        self.graph = detection_graph
        self.cap = cv2.VideoCapture(self.video_file)
        self.FPS = self.cap.get(5)
        self.w, self.h = self.cap.get(3), self.cap.get(4)
        self.FRAME_COUNT = self.cap.get(7)
        self.VIDEO_LENGTH = int(self.FRAME_COUNT/self.FPS)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
    
        
    def play(self, start_frame=1):
        """
        Playing the video inside a player
        Arguments:
            start_frame: int, the starting frame
        """
        with self.graph.as_default():
            with tf.Session(graph=self.graph) as self.sess:
                ind = start_frame
                animation = Player(self.fig, self._update, dis_start=1, dis_stop=50)#self.FRAME_COUNT)
                plt.show()

    
    def _add_detected_objects(self, image_np):
        """
        add the detected objects by the model to the image
        Arguments:
            image_np: 2d numpy array, image, i.e., a frame of the video
        """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        return image_np
    
    def _update(self, ind):
        """
        update object for the player
        Arguments:
            ind: int, index of the frame
        """
        self.cap.set(1, ind) 
        self.ax.clear()
        self.fig.title('Time (sec):', ind/self.FPS)
        ret, image_np = self.cap.read()         
        image_np = self._add_detected_objects(image_np)
        self.ax.imshow(cv2.resize(image_np, (800,600))) 

    
    
#   ---------------------------------------
if __name__ == '__main__':

    # What model to download.
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90

    #   ---------------------------------
    # Download Model

    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())

    #   ---------------------------------
    # Load a (frozen) Tensorflow model into memory.

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    #   ---------------------------------
    # Loading label map
    # Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)


    #   ---------------------------------
    # Helper code

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    #   ---------------------------------
    # Detection

    # For the sake of simplicity we will use only 2 images:
    # image1.jpg
    # image2.jpg
    # If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
    #PATH_TO_TEST_IMAGES_DIR = 'test_images'
    #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

    # Size, in inches, of the output images.
    #IMAGE_SIZE = (12, 8)

    video_file = './data/city.mp4'

    vod = VideoObjectDetection(video_file, detection_graph)
    vod.play(1)
