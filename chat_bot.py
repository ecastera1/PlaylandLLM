#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 12:11:36 2023

@author: Enrique Casterá
"""

from enum import Enum

class ChatSubject(Enum):
    PROMPT = 0,
    HUMAN = 1,
    AI = 2,
    

class ChatElement:
    def __init__(self, subject, text):
        self.subject = subject
        self.text = text

class ChatBot:
    history=[]
    chat_history_max_length=1024
    
    def __init__(self, prompt, human_name, bot_name, chat_history_max_length):   
        self.history = []
        self.human_name = human_name
        self.bot_name = bot_name
        self.history.append(ChatElement(ChatSubject.PROMPT,prompt))
        self.chat_history_max_length = chat_history_max_length
    
    def append(self, subject,text):
        self.history.append(ChatElement(subject,text))
        
    def append_human(self, text):
        self.history.append(ChatElement(ChatSubject.HUMAN,text))
    
    def append_ai(self, text):
        self.history.append(ChatElement(ChatSubject.AI,text))
            
    def get_item_str_index(self, index):            
        return self.get_item_str(self.history[index])
    
    def get_item_str(self, chat_element):        
        
        if chat_element.subject == ChatSubject.PROMPT:
            return chat_element.text
        
        sname=""
        if chat_element.subject == ChatSubject.AI:
            sname = self.bot_name
        if chat_element.subject == ChatSubject.HUMAN:
            sname = self.human_name        
        s=f"{sname}: {chat_element.text}"
        
        return s
    
    def get_history_list(self):
        return self.history
    
    def get_history_str(self, full_history=False):
        if full_history:
            str_history = ""
            for item in self.history:
                s = self.get_item_str(item)
                str_history = str_history + s + "\n"            
            return str_history
        
        # return str = initial_prompt + history[x..length] 
        # where x is calculated so len(str) don't overflow chat_history_max_length
        
        # initial prompt
        str_history = self.get_item_str_index(0)
        
        index=len(self.history)-1
        length=len(str_history)
        while index>0:
            s = self.get_item_str_index(index)
            length += len(s)
            if length>self.chat_history_max_length:
                break
            index -= 1
            
        # index = 1 if index==0 else index
        index = index + 1
        #print(f"max_length calculations {length} range from {index} to {len(self.history)}")
        
        # from index to history length
        for i in range(index,len(self.history)):            
            s = self.get_item_str_index(i)
            str_history = str_history + s + "\n"    
            
        # trim it always to max_length
        str_history = str_history[-self.chat_history_max_length::]        
        return str_history

    