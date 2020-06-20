#!/usr/bin/env python3
import os
import sys
import logging as log
from openvino.inference_engine import IENetwork, IECore

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        self.plugin = IECore()
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.net_plugin = None
        self.infer_request_handle = None
        
    def load_model(self, model, device="CPU", cpu_extension=None):
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
            
        self.network = IENetwork(model=model_xml, weights=model_bin)
        self.net_plugin = self.plugin.load_network(self.network, device)
        
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))
        return

    def get_input_shape(self):
        return self.network.inputs[self.input_blob].shape

    def exec_net(self, image, request_id):
        self.infer_request_handle = self.net_plugin.start_async
        self.infer_request_handle(request_id = request_id, 
            inputs={self.input_blob: image})
        return

    def wait(self):
        status = self.net_plugin.requests[0].wait(-1)
        return status

    def get_output(self):
        return self.net_plugin.requests[0].outputs[self.output_blob]
