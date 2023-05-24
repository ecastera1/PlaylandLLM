#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Command prompt module with TAB autocompletion for interactive commands

Created on Mon Sep  5 15:35:30 2022
@author: Enrique Castera 
"""

import readline
import os
import atexit
import glob

HISTORY_FILE = "./history.txt"
HISTORY_LENGTH = 20

#
# populate .txt files with prompts for completion
#
PROMPT_FOLDER="./prompts"
VALID_FILES = [ os.path.basename(f) 
               for f in list(glob.glob(PROMPT_FOLDER+"/*.txt", recursive=False))
               ]



def get_valid_files():
    global PROMPT_FOLDER
        


SINGLE_COMMAND = {
    '<intro>':  'run single next inference',
    '<number':  'run %d inferences, like loop %d',
    'quit':     'exit to OS',
    'go':       'run single next inference',
    'clean':    'delete output path log files' ,
    'help':     'display commands help',
    'save_model':   'save model pytorch files to local folder',
    'chat':     'chat mode HUMAN with AI',
    }


STR_COMMAND = {
    'from_file':    'read single prompt from a text file',
    'list_from_file': 'read a list of prompts from a text file, one prompt per line',
    'model':        'load model from valid models',
    'lang':         'switch language en, es'
    }

INT_COMMAND = {
        'loop':     'repeat %d inferences',
        'autochat': 'chat mode automatic AI with AI, repeat %d times',        
        'storychat': 'chat mode automatic AI with AI alternating roles, repeat %d times',        
        'seed':     'set random seed to %d',
        'top_k':    'top_k %d generation parameter',
        'no_repeat_ngram_size': 'generation parameter',
        'max_length':   'max length %d of response in tokens, generation parameter',
        'max_time':     'max_time %d of response, generation parameter',
        'chat_history_max_length':  'max length of chat history memory',
        'num_return_sequences': '%d generation parameter',
        'audio':    'audio 1 on, 0 off'
   }

FLOAT_COMMAND = {
    'temperature': 'temperature %f, generation parameter',
    'top_p': 'top_p %f, generation parameter'
    }

VALID_COMMANDS_DICT = SINGLE_COMMAND | STR_COMMAND | INT_COMMAND | FLOAT_COMMAND 
VALID_COMMANDS = list(VALID_COMMANDS_DICT.keys())


COMPLETION_LIST = VALID_COMMANDS + VALID_FILES
#print(f"COMPLETION_LIST {COMPLETION_LIST}")


def get_valid_commands():
    return str(VALID_COMMANDS)

def print_commands_help():
    print(f"Command help:\n{VALID_COMMANDS_DICT}")
    

def completer(text, state):
    options = [i for i in COMPLETION_LIST if i.startswith(text)]

    if state < len(options):
        return options[state]
    else:
        return None


def init_command_prompt():
    if not os.path.exists(HISTORY_FILE):
        open(HISTORY_FILE, 'a+').close()

    readline.read_history_file(HISTORY_FILE)
    readline.set_history_length(HISTORY_LENGTH)
    atexit.register(readline.write_history_file, HISTORY_FILE)

    readline.set_completer(completer)
    for line in (
        "tab: complete",
        "set show-all-if-unmodified on",
            "set enable-keypad on"):
        readline.parse_and_bind(line)


def command_prompt():
    #print("### Enter command, TAB, help, quit, a number or enter to continue:")
   
    # using readline with autocomplete
    user_input = input("> ")

    if len(user_input) == 0:
        return ('next', 0)

    cmd_list = user_input.split()
    # print("### cmd "+str(cmd_list))
    command = cmd_list[0]

    # special commands
    if command.isnumeric():
        arg0 = int(cmd_list[0])
        print(f"command loop {arg0}")
        return ('loop', arg0)

    if (command == 'prompt'):
        cmd_list.pop(0)
        arg0 = str(' '.join(cmd_list))
        print(f"command prompt '{arg0}'")
        return ('prompt', arg0)

    # parameter commands    
    if command in SINGLE_COMMAND.keys():
        print(f"single_command {command}")
        return (command, 0)
    
    if len(cmd_list)==1:
        print("### ERROR command needs at least 1 argument")
        return None
    
    if command in STR_COMMAND.keys():
        arg0 = str(cmd_list[1])
        print(f"str_command {command} {arg0}")
        return (command, arg0)

    if command in INT_COMMAND.keys():        
        arg0 = int(cmd_list[1])
        print(f"int_command {command} {arg0}")
        return (command, arg0)
    
    if command in FLOAT_COMMAND.keys():
        arg0 = float(cmd_list[1])
        print(f"float_command {command} {arg0}")
        return (command, arg0)

    print("### ERROR Invalid command_prompt")
    return None


print("### Initializing command_prompt...")
init_command_prompt()
print("### Done")
