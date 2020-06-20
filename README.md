# People Counter App at the Edge

![people-counter-python](./images/people-counter-image.png)

## Description

The app will detect people in a designated area, providing the number of people in the frame, average duration of people in frame, and total count.

## How it Works

The counter will use the Inference Engine included in the Intel® Distribution of OpenVINO™ Toolkit. The model used is able to identify people in a video frame. The app count the number of people in the current frame, the duration that a person is in the frame (time elapsed between entering and exiting a frame) and the total count of people. It then sends the data to a local web server using the Paho MQTT Python package.


![architectural diagram](./images/arch_diagram.png)
        
## Setup

Follow the instructions in setup.md to install:

1. Intel® Distribution of OpenVINO™ toolkit
2. Nodejs and its dependencies
3. Python and npm modules

There are three components that need to be running in separate terminals for this application to work:

-   MQTT Mosca server 
-   Node.js* Web server
-   FFmpeg server
     
From the main directory:

* For MQTT/Mosca server:
   ```
   cd webservice/server
   npm install
   ```

* For Web server:
  ```
  cd ../ui
  npm install
  ```
  **Note:** If any configuration errors occur in mosca server or Web server while using **npm install**, use the below commands:
   ```
   sudo npm install npm -g 
   rm -rf node_modules
   npm cache clean
   npm config set registry "http://registry.npmjs.org"
   npm install
   ```
### Preparing the machine learning model

In order to test the model please follow the next steps:

1. Create a "models/caffe_model/" directory

2. Access from terminal: cd models/caffe_model

3. Run in terminal: git clone https://github.com/zlingkang/mobilenet_ssd_pedestrian_detection

4. Access to model folder: cd mobilenet_ssd_pedestrian_detection

5. The model conversion to IR were done with the following command:
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model MobileNetSSD_deploy10695.caffemodel --input_proto MobileNetSSD_deploy.prototxt

   It could be recommended to upgrade protobuf:
```
   [ WARNING ]  
   Detected not satisfied dependencies:
         protobuf: installed: 3.5.1, required: 3.6.1
```
6. To upgrade protobuf run on terminal: 

```
sudo pip3 install --upgrade protobuf==3.6.1
```
    

## Running the App

### Step 1 - Start the Mosca server

```
cd webservice/server/node-server
node ./server.js
```

You should see the following message, if successful:
```
Mosca server started.
```

### Step 2 - Start the GUI

Open new terminal and run below commands.
```
cd webservice/ui
npm run dev
```

You should see the following message in the terminal.
```
webpack: Compiled successfully
```

### Step 3 - FFmpeg Server

Open new terminal and run the below commands.
```
sudo ffserver -f ./ffmpeg/server.conf
```

### Step 4 - Run the code

Open a new terminal to run the code. 

You must configure the environment to use the Intel® Distribution of OpenVINO™ toolkit one time per session by running the following command:

```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

You should also be able to run the application with Python 3.6, although newer versions of Python will not work with the app.

#### Command examples

After running the required servers and setting the enviroment, the project can be tested with the following commands;

1. Video input

```
python main_caffe.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m models/caffe_model/mobilenet_ssd_pedestrian_detection/MobileNetSSD_deploy10695.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.38 -f 24| ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

2. Single image input:

```
python main_caffe.py -i resources/person.PNG -m models/caffe_model/mobilenet_ssd_pedestrian_detection/MobileNetSSD_deploy10695.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.40 
```

The output image is saved in the same directory.
