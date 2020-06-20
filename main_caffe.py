import os
import os.path as op
import sys
import time
import numpy as np
import socket
import json
import cv2

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    parser.add_argument("-f", "--framerate", type=int, default=24,
                       help="Video framerate, required for time calculation"
                       "(24 by default)")
    return parser


def connect_mqtt():
    client = mqtt.Client()
    return client

def get_data(result, threshold ,frame_w, frame_h):
    data_output = result[0][0]
    filtered_boxes = []
    
    for obj in data_output:
        if obj[2] > threshold:
            xmin = int(obj[3] * frame_w)
            ymin = int(obj[4] * frame_h)
            xmax = int(obj[5] * frame_w)
            ymax = int(obj[6] * frame_h)
            filtered_boxes.append([xmin, ymin, xmax, ymax])
            
    return filtered_boxes

def draw_boxes(frame, box_list):
    color = (255, 0, 0) 
    thickness = 2
    for box in box_list:
        cv2.rectangle(frame, (box[0],box[1]), (box[2], box[3]), color, thickness)
        cv2.putText(frame, 'Person', (box[0]+2,box[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 255, 0), 1)

def validate_count(count_value, pila):
    flag = True    
    
    for i in pila:
        if i != count_value:
            flag = False
            break
    
    return flag

def validate_extension(file_format):
    extension = op.splitext(file_format)[1]
    
    video_format_allowed = [".mp4"]
    image_format_allowed = ['.PNG', ".jpg"]
    
    if extension in video_format_allowed:
        return "video"
    elif extension in image_format_allowed:
        return "image"
    else:
        return "Not_Allowed"
        
def infer_on_stream(args, client, extension):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    
    frame_rate = args.framerate
    
    # Initialise the class
    infer_network = Network()
    
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    infer_network.load_model(args.model, args.device, args.cpu_extension)
    net_input_shape = infer_network.get_input_shape()
    
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    
    #Values to filter detection signals
    total_count = 0
    previous_state = 0
    pila = [0]*24
    
    #Values to calculate duration
    duration_acc = 0.0
    duration_avg = 0.0
    start_time = 0.0
    frames_counted = 0
    
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        key = cv2.waitKey(60)
        
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        
        #Model transformations
        p_frame = p_frame - 127.5
        p_frame = p_frame * 0.003743
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        infer_network.exec_net(p_frame, 0)
        
        if infer_network.wait() == 0:
            result = infer_network.get_output()
            boxes = get_data(result, args.prob_threshold, frame_width, frame_height)
            
            if extension == "video":
                count = len(boxes)
                pila.append(count)
                pila.pop(0)       

                if validate_count(count, pila):
                    current_count = count
                    if (current_count != 0) and (start_time == 0.0) :
                        start_time = time.time()

                ### current_count, total_count and duration to the MQTT server ###
                ### Topic "person": keys of "count" and "total" ###
                ### Topic "person/duration": key of "duration" ###
                
                #client.publish("person", json.dumps({"count": current_count, "total":total_count}))
                client.publish("person", json.dumps({"count": current_count}))

                if (current_count != previous_state):
                    client.publish("person", json.dumps({"total":total_count}))
                    total_count+= current_count

                    if (current_count == 0) and (previous_state > 0):
                        duration_acc += time.time() - start_time
                        start_time = 0.0
                        duration_avg = duration_acc / total_count
                        #duration_acc = frames_counted / frame_rate
                        #duration_avg = duration_acc / total_count
                        
                    if (current_count != 0):
                        client.publish("person/duration", json.dumps({"duration":duration_avg}))
        
        if extension == "video":
            previous_state = current_count
            
            draw_boxes(frame, boxes)
            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()
            
            
        if extension == "image":
            draw_boxes(frame, boxes)
            cv2.imwrite("resources/person_detected.jpg",frame)
        
        
        
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
        
def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    #Validate input extension
    extension = validate_extension(args.input)
    if extension != "Not_Allowed":
        # Perform inference on the input stream
        infer_on_stream(args, client, extension)
    else:
        print("Input file extension not allowed")

if __name__ == '__main__':
    main()
