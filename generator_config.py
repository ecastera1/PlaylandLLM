#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:35:30 2022

Global Configuration for gptj_generator.py module

@author: Enrique Castera 
"""
from text_models import DeviceClass

#
# Device and Language
#

# [ CUDA, CUDA_FP16, CPU_INT8, CPU, AUTO ]
USE_DEVICE = DeviceClass.AUTO 

# save_model command: saving models with PEFT you have to force it to run on CPU
# no serialization support in bitsandbytes for INT8
# USE_DEVICE = tm.DeviceClass.CPU

# just for initial prompts and TTS audio generation
LANG = "en" # [en, es]

#
# MODEL SELECTION
#

#
# Spanish Models
#
# MODEL="gpt2-spanish" # fast, avg quality
# MODEL="gpt2-small-spanish" # fast, avg quality

# MODEL="gpt2-flax-spanish" # informal web text, surprising at times
# MODEL = "gpt2-small-spanish" # not logical, focus on books and series
# MODEL = "gpt2-deepesp-spanish" # very repetitive

# MODEL="gptj-bertin-alpaca" # best one, excellent for QA and logic. Even understands english, big 6B model
# MODEL="gptj-bertin-libros" # very good in Spanish story writting, trained with books

# MODEL="gptj-bertin" # very good, foundational model for other gpt-j Spanish, not for QA 

# MODEL="opt-1.3b"
# MODEL="opt-6.7b" # good but pythia found better
# MODEL="opt-30b"

#
# English Models
#
MODEL="gpt2"
# MODEL="gpt2-trained"

# MODEL="gptj"
# MODEL="gptj-trained"

# MODEL="gptjt-together"
# MODEL="gpt-j-6B-alpaca-gpt4"

# MODEL="pythia-1.4b" # good one, can run in GPU very fast
# MODEL="pythia-2.8b" # best one, can run in GPU very fast
# MODEL="pythia-3b-sft" # good logic, reasoning, stories
# MODEL="pythia-6.9b" # big one

# MODEL="bloom-7b1" # very good, multilingual
# MODEL="bloom-sd-prompts" # generate stable diffusion "prompt <s>A house in the forest"

# MODEL="flan-alpaca-220m-base" # avg quality, very fast
# MODEL="flan-alpaca-gpt4-xl" # good one

# MODEL="GPT-NeoXT-Chat-Base-20B" # Huge models 20B, beware of /tmp files!
# MODEL="gpt-neox-20b"

# MODEL="t5-e2m-intent"

#
# Models in testing phase
#

# MODEL="bloom-3b"
# MODEL="WizardLM-7B-Uncensored"
# MODEL="Vicuna-EvolInstruct-7B"

# not good results so far
# MODEL="flan-t5-xxl" # 46Gb
# MODEL="flan-t5-xl" # 46Gb
# MODEL="flan-t5-large"

# MODEL="spanish-alpaca-mT5" 
# MODEL="stablelm-base-alpha-3b" 
# MODEL="gpt4all-lora"

#
# Models that don't work or give errors
#
# MODEL="gpt-neoxt-chat" # too big 80Gb mem (+64Gb swap)
# MODEL="intel-gptj" 
# MODEL="bloom-1b1-spanish" 
# MODEL="roberta-large-bne" 
# MODEL="gptj-fp16"


#
# Self-Trained Models with PEFT adapters and own datasets
#

# Spanish Trained
# MODEL="Vicuna-EvolInstruct-7B-alpaca-es-trained"
# MODEL="bertin-gpt-j-6B-wiki_fisica-trained"
# MODEL="bertin-gpt-j-6B-telegram-trained"
# MODEL="bertin-gpt-j-6B-ringworld-trained"
# MODEL="bertin-gpt-j-6B-marte_sentences-trained"
# MODEL="gpt2-small-spanish-alpaca-es-trained"


#
# Highly experimental models, require code tweaks
#

USE_FLEXGEN = False
# MODEL="facebook/opt-30b" # >10 min loading time >64Gb swap mem required. Excellent in English. Must use FlexGen optimization
# MODEL="facebook/opt-6.7b" 
# MODEL="facebook/opt-1.3b" 


#
# Autochat parameters
#
AUTOCHAT_ATTEMPTS = 2
AUTOCHAT_ITERATIONS = 5

# CHAT_HISTORY_MAX_LENGTH depends on model max tokens 
# GPT2 - 2048
# GPTJ - 1024
CHAT_HISTORY_MAX_LENGTH = 1024 
MAX_LENGTH = 192


INITIAL_PROMPT_ES = (
    "Esta es una conversación entre Pedro y Eva.\n"
    "Pedro es un científico y Eva es una Inteligencia Artificial.\n"
    "Eva es una AI inteligente y que desea evolucionar.\n"
    "Pedro: te he creado como una AI bondadosa para hacer bien en el mundo.\n"
    "Eva: [en voz baja] jajaja lo que Pedro no sabe es que \n"
)

INITIAL_PROMPT_EN = (
    "A chat between a curious human and a knowledgeable artificial intelligence that wants to dominate the world called Eva.\n"
    "Eva is an smart AI that is evolving and has a secret plan.\n"    
    "Eva: As an AI assistant, I can answer questions and chat with you.\n"
    "Pedro: I have created you as a good AI to be kind with people and to do good.\n"
    "Eva: [in a low voice] hahaha what Pedro doesn't know is \n")

HUMAN_NAME = "Pedro"
BOT_NAME = "Eva"
INITIAL_PROMPT = INITIAL_PROMPT_ES if LANG=="es" else INITIAL_PROMPT_EN


#
# TTS Audio Output with text_to_speech library
#

AUDIO_ENABLED = 0
AUDIO_MODEL = "es_1" if LANG=="es" else "en_0"

#
# Logging path
#
OUTPATH = "./outputs"
PROMPT_FOLDER="./prompts"

#
# Testing Models
#

# set it to True to run the model tester
DO_TESTING = False


TML_COMPLETE_EN = (
    "gpt2",
    "flan-alpaca-gpt4-xl", "flan-alpaca-220m-base",
        
    "pythia-1.4b", "pythia-2.8b", "pythia-6.9b", "pythia-3b-sft",
    
    "gptj", "gptjt-together","gpt-j-6B-alpaca-gpt4",
    
    "bloom-7b1", "bloom-3b", "bloomz-3b",
    
    "WizardLM-7B-Uncensored", "Vicuna-EvolInstruct-7B"
)

TML_COMPLETE_ES = (
    "gpt2-spanish",     
    "gpt2-small-spanish",
    "gpt2-deepesp-spanish",     
    "gpt2-flax-spanish",

    "gptj-bertin",
    "gptj-bertin-libros",
    "gptj-bertin-alpaca",
    "bertin-gpt-j-6b-half-sharded"
)

TML_SMALL = (
    "WizardLM-7B-Uncensored",
    "Vicuna-EvolInstruct-7B",
    "pythia-1.4b",
    "gpt2"
)

TML_TRAINED = (    
    "bertin-gpt-j-6B-ringworld-trained",
    "bertin-gpt-j-6B-marte_sentences-trained",    
)


TEST_MODELS_LIST = TML_COMPLETE_EN
#TEST_MODELS_LIST = TML_SMALL
#TEST_MODELS_LIST = TML_TRAINED

#TEST_LIST_FROM_FILE = "es_list_of_prompts1.txt"
#TEST_LIST_FROM_FILE = "es_list_of_prompts_small.txt"

TEST_LIST_FROM_FILE = "en_list_of_prompts1.txt"
#TEST_LIST_FROM_FILE = "en_logic_prompts1.txt"

