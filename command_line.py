#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 15:52:03 2023

@author: quique
"""
import argparse

import text_models as tm

     
def read_command_line():    
    parser = argparse.ArgumentParser()

    valid_devices = list(tm.DeviceClass)
    valid_devices = [str(x).split('.')[1] for x in valid_devices]
    
    valid_models = tm.get_available_models()
     
    parser.add_argument(
        "--model",
        type=str,
        choices=valid_models,
        help="model to use",        
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=valid_devices,        
        help="device to use",
        #default="AUTO"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        #default="light is always",
        help="initial prompt to generate text"
    )  
    parser.add_argument(
        "--testing", 
        action='store_true',         
        help="testing mode"
    )
    parser.add_argument(
        "--test_list_from_file",
        type=str,        
        help="text file containing prompts for testing",        
    )
    parser.add_argument(
        "--test_models",
        type=str, 
        nargs="?",
        help="list of text models for testing",        
    )    
    
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":    
    opt = read_command_line()
    print(opt)
