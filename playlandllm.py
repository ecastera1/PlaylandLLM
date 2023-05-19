#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:35:30 2022
@author: Enrique Castera 
"""

import os
import time
import torch
import sys
import tqdm
import glob
import traceback
import random
from pathlib import Path
from transformers import set_seed

import text_to_speech as tts
import text_models as tm
import tests_results as tr
import command_prompt
import command_line

from termcolor import colored

# GLOBAL config variables
from generator_config import *

# color output
COLOR_DEBUG="magenta"
COLOR_AI="light_yellow"
COLOR_PROMPT="green"
COLOR_HUMAN = "light_blue"



my_stopping_criteria = None

def make_outpath():
    global OUTPATH
    global DO_TESTING
    global MODEL

    x = "./outputs"
    if DO_TESTING:
        x = x + "-" + MODEL

    if os.path.isdir(x) == False:
        os.mkdir(x)
        print(f"Directory {x} created")

    OUTPATH = x
    print(f"OUTPATH directory {OUTPATH}")


def clean_output_dir():
    global OUTPATH
    files = glob.glob(OUTPATH + "/*.txt", recursive=False)
    for f in files:
        try:
            # print(f)
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))


def read_prompt_from_file(from_file, parse_as_list=False):
    if os.path.isfile(from_file):
        with open(from_file, "r") as f:
            if parse_as_list:
                data = f.read().splitlines()
            else:
                data = f.read()

        if parse_as_list:
            result = []
            for x in data:
                if len(x) > 0:
                    result.append(x.strip())
            return result

        return data
    else:
        return ""


def generate_unique_filename(prompt, seed):
    non_valid = list(" .-,?!'\"")
    s = prompt[:20]
    for c in non_valid:
        s = s.replace(c, '_')
    h = hex(hash(prompt))[-6::]
    name = s + "_" + h + "_" + str(seed) + ".txt"
    return name


def generate_list_of_prompts(modelwrapper, l_prompts, opt):
    global my_stopping_criteria
    global AUDIO_ENABLED
    global AUDIO_MODEL

    for p in tqdm.tqdm(l_prompts):
        if len(p) == 0:
            continue
        print(colored(f"### Generating prompt {p}", COLOR_PROMPT))
        opt['prompt'] = p
        outputs = []

        tic = time.time()
        # outputs=["ejemplo"]
        if USE_FLEXGEN:
            text_output = tm.flexgen_generate(
                modelwrapper,
                p,
                temperature=opt['temperature'],
                max_length=opt['max_length'])
            outputs = [text_output]
        else:
            outputs = modelwrapper.generator(p,
                                             temperature=opt['temperature'],
                                             do_sample=True,
                                             # num_beams=2,
                                             top_p=opt['top_p'],
                                             top_k=opt['top_k'],
                                             no_repeat_ngram_size=opt['no_repeat_ngram_size'],
                                             max_length=opt['max_length'],
                                             max_time=opt['max_time'],
                                             num_return_sequences=opt['num_return_sequences'],
                                             # end_sequence="###" # stopping
                                             # sequence for generation
                                             stopping_criteria=my_stopping_criteria,
                                             )
            outputs = [(x['generated_text']) for x in outputs]

        toc = time.time()
        elapsed = toc - tic

        print("### Generated text:\n")
        
        sep = "-" * 20        
        print(colored(sep,COLOR_AI))
        for text in outputs:            
            #print(colored(f"### output {index}", COLOR_AI))
            print(colored(text, COLOR_AI))             
            
        print(colored(sep,COLOR_AI))

        print(colored(f'### Elapsed: {elapsed:.2f}s', COLOR_DEBUG))
        fname = generate_unique_filename(p, opt['seed'])
        outfile = os.path.join(OUTPATH, fname)

        with open(outfile, "w", encoding="utf-8") as f:
            index = 1
            for text in outputs:
                f.write(sep + "\n")
                f.write(f"### output {index}\n")
                f.write(text + "\n")
                f.write(sep + "\n")
                index += 1
            f.write("opt = " + str(opt) + "\n")
            f.write("prompt = " + opt['prompt'] + "\n")
            f.write(f"Memory {modelwrapper.model.get_memory_footprint()/1024/1024:.2f}Mb CUDA Memory: {torch.cuda.memory_allocated()/1024/1024:.2f}Mb\n")
            f.write(f'Elapsed time = {elapsed:.2f}s\n')


        # log all prompts in a single file
        fname = generate_unique_filename("prompts_", 0)
        outfile = os.path.join(OUTPATH, fname)
        with open(outfile, "a", encoding="utf-8") as f:
            for text in outputs:
                f.write(text + "\n")

        # play audio of all texts
        if AUDIO_ENABLED == 1:
            for text in outputs:
                tts.tts_play(AUDIO_MODEL, text)

    return outputs


def add_extra_tokens(buffer):
    global LANG
    if LANG == "es":
        l_tokens = [
            'y',
            'por tanto',
            'entonces',
            'ademas',
            'por lo que',
            'por consiguiente',
            'entonces',
            'de lo que se deduce']
    else:
        l_tokens = ['and', 'therefore', 'then', 'so that', 'and then']

    token = random.choice(l_tokens)
    return buffer + " " + token + " "


def chat_bot(modelwrapper, opt, chat_type):
    global CHAT_HISTORY_MAX_LENGTH, USE_FLEXGEN
    global my_stopping_criteria
    global AUDIO_MODEL, AUDIO_ENABLED, AUTOCHAT_ITERATIONS, AUTOCHAT_ATTEMPTS
    global HUMAN_NAME, BOT_NAME

    history = []

    history.append(opt['prompt'])
    tic = time.time()
    set_seed(opt['seed'])

    print(colored("Enter . to reset chat. Empty input to exit chat mode", COLOR_PROMPT))
    print(colored(f">> INITIAL_PROMPT {history[0]}", COLOR_PROMPT))

    if chat_type == 0:
        iterator = tqdm.tqdm(range(AUTOCHAT_ITERATIONS))
    else:
        iterator = range(100)

    for step in iterator:

        # max_length
        #outputs = model.generate(user_input, max_new_tokens=HISTORY_MAX_LENGTH, pad_token_id=tokenizer.eos_token_id)

        if chat_type == 1:
            inp = input(colored(f">> {step} {HUMAN_NAME}: ",COLOR_HUMAN))
            user_input = f"{HUMAN_NAME}: {inp.strip()}"
            
            if not inp:
                print("exit...")
                break

            if inp == '.':
                print("Resetting")
                history = [opt['prompt']]
                print(colored(f">> INITIAL_PROMPT {history[0]}", COLOR_PROMPT))
                continue

            history.append(user_input)

        # FIX: calc history based on sentences or tokens
        str_history = "\n".join(history)
        buffer = str_history[-CHAT_HISTORY_MAX_LENGTH::].strip()

        if len(buffer) == 0:
            break

        #print(f"\n>> {step} Prompt: len {len(buffer)} CHAT_HISTORY_MAX_LENGTH {CHAT_HISTORY_MAX_LENGTH}\n### {buffer} \n###")

        attempts = 0
        response = ""
        response_without_prompt = ""

        while attempts < AUTOCHAT_ATTEMPTS:
            if USE_FLEXGEN:
                response_without_prompt = tm.flexgen_generate(
                    modelwrapper,
                    buffer,
                    temperature=opt['temperature'],
                    max_length=opt['max_length'])
            else:
                outputs = modelwrapper.generator(buffer,
                                                 temperature=opt['temperature'],
                                                 do_sample=True,
                                                 # num_beams=2,
                                                 top_p=opt['top_p'],
                                                 top_k=opt['top_k'],
                                                 no_repeat_ngram_size=opt['no_repeat_ngram_size'],
                                                 # max_length=opt['max_length'],
                                                 max_new_tokens=opt['max_length'],
                                                 max_time=opt['max_time'],
                                                 num_return_sequences=opt['num_return_sequences'],
                                                 # end_sequence="###" #
                                                 # stopping sequence for
                                                 # generation
                                                 stopping_criteria=my_stopping_criteria,
                                                 )
                outputs = [(x['generated_text']) for x in outputs]
                response = outputs[0]
                response_without_prompt = response[len(buffer)::].strip()

            if len(response_without_prompt) > 0:
                break

            attempts = attempts + 1
            buffer = add_extra_tokens(buffer)
            print(colored(f"### empty response retrying {attempts} with new buffer {buffer}",COLOR_DEBUG))

        if chat_type == 1:
            print(colored(f"#{step}: {user_input}", COLOR_HUMAN))
            
        print(colored(f"#{step}: {BOT_NAME}: {response_without_prompt}", COLOR_AI))
        history.append(response_without_prompt)

        save_bot_chat(history, opt, 0)
        opt['seed'] = opt['seed'] + 1
        set_seed(opt['seed'])

        if AUDIO_ENABLED == 1:
            tts.tts_play(AUDIO_MODEL, response_without_prompt)

    toc = time.time()
    elapsed = toc - tic
    print(colored("#### chat_bot DONE:\n", COLOR_DEBUG))
    
    for idx, x in enumerate(history):        
        if idx==0:
            c=COLOR_PROMPT
        else:
            c= COLOR_AI if (idx % 2)==0 else COLOR_HUMAN
            
        print(colored(f"#{idx}: {x}",c))
        
    print(colored(f'elapsed time = {elapsed:.2f}s\n', COLOR_DEBUG))
    save_bot_chat(history, opt, elapsed)
    return


def save_bot_chat(history, opt, elapsed=0):
    global OUTPATH
    if elapsed > 0:
        fname = "bot_" + generate_unique_filename(opt['prompt'], opt['seed'])
    else:
        fname = "bot_" + generate_unique_filename(opt['prompt'], 0)

    outfile = os.path.join(OUTPATH, fname)
    sep = "-" * 20
    with open(outfile, "w", encoding="utf-8") as f:
        index = 1
        for text in history:
            f.write(sep + "\n")
            f.write(f"### output {index}\n")
            f.write(text + "\n")
            f.write(sep + "\n")
            index += 1

        f.write("opt = " + str(opt) + "\n")
        f.write("prompt = " + opt['prompt'] + "\n")
        if elapsed > 0:
            f.write(f'elapsed time = {elapsed:.2f}s\n')


def free_memory(to_delete: list):
    import gc
    import inspect
    calling_namespace = inspect.currentframe().f_back

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        del _var

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    gc.collect()


def test_models():
    global USE_DEVICE
    global MAX_LENGTH
    global AUDIO_ENABLED
    global MODEL
    global OUTPATH
    global INITIAL_PROMPT
    global PROMPT_FOLDER
    global TEST_LIST_FROM_FILE
    global TEST_MODELS_LIST


    # check input parameters
    f = PROMPT_FOLDER + "/" + TEST_LIST_FROM_FILE
    if os.path.isfile(f) == False:
        print(f"\nERROR {f} file not found for prompt testing. Check value of TEST_LIST_FROM_FILE")
        return
    else:
        print(f"TEST_LIST_FROM_FILE {f} found!")
        
    available_models = tm.get_available_models()    
    for x in TEST_MODELS_LIST:
        if (x in available_models) == False:
            print(f"\nERROR '{x}' model not found in list of available models. Check TEST_MODELS_LIST.\nAvailable models: {available_models}")
            return
        else:
            print(f"Will test model '{x}'")
    

    for my_model in tqdm.tqdm(TEST_MODELS_LIST):
        MODEL = my_model
        print(f'\n\n### Testing model: {MODEL}')

        tic = time.time()
        # force CPU, GPT-J model requires 256MB GPU!!
        if USE_FLEXGEN:
            modelwrapper = tm.flexgen_load_model(MODEL, USE_DEVICE)
        else:
            modelwrapper = tm.load_model(MODEL, USE_DEVICE)

        toc = time.time()
        elapsed = toc - tic
        print(colored(f'### Elapsed: {elapsed:.2f}s', COLOR_DEBUG))

        make_outpath()
        print(f"### clean output {OUTPATH}\n")
        clean_output_dir()

        seed = int(time.time())

        prompt = INITIAL_PROMPT
        opt = {
            "model_name": modelwrapper.name,
            "device": modelwrapper.use_device,
            "seed": seed,
            "prompt": "",
            "from_file": "",
            "list_from_file": "",
            "temperature": 0.4,
            "top_p": 1.0,
            "top_k": 50,
            "no_repeat_ngram_size": 3,
            "max_length": MAX_LENGTH,
            "max_time": 300.0,
            "num_return_sequences": 1
        }

        opt['prompt'] = prompt
        opt['list_from_file'] = PROMPT_FOLDER + "/" + TEST_LIST_FROM_FILE
        opt['from_file'] = ""

        from_file = opt['list_from_file']
        print("### reading list of prompts from file " + from_file)
        x = read_prompt_from_file(from_file, parse_as_list=True)
        if len(x) > 0:
            l_prompts = x
        else:
            print("### ERROR no prompts to generate!!!")
            return

        tic = time.time()

        try:
            outputs = generate_list_of_prompts(modelwrapper, l_prompts, opt)
            seed = seed + 1
            opt['seed'] = seed

        except KeyboardInterrupt:
            print('*** User abort')
            loop_cnt = 0
            chat_loop_cnt = 0
            pass
        except Exception as ex:
            print("!!! EXCEPTION " + str(ex))
            traceback.print_exc()
            loop_cnt = 0
            chat_loop_cnt = 0
            pass

        toc = time.time()
        elapsed = toc - tic
        print(colored(f"### FINISH testing {MODEL} Elapsed: {elapsed:.2f}s", COLOR_DEBUG))

        # compute KPIs for the model
        total_length_inputs = sum(len(x) for x in l_prompts)
        total_length_outputs = sum(len(x) for x in outputs)
        total_length = total_length_inputs + total_length_outputs
        avg_chars_per_sec = float(total_length) / float(elapsed)
        avg_response_length = float(
            total_length_outputs) / float(len(l_prompts))
        tr.add_test_result(
            modelwrapper.name,
            modelwrapper.use_device,
            elapsed,
            avg_chars_per_sec,
            avg_response_length)

        modelwrapper = None
        free_memory([modelwrapper])

        # save KPIs intermediate
        CSV_FILE = "results.csv"
        PICKLE_FILE = "results.pkl"
        tr.save_results(CSV_FILE, PICKLE_FILE)

    print("Done testing")


def switch_model(modelwrapper, my_model):
    
    valid_models = tm.get_available_models()
    if (my_model in valid_models)==False:
        print(f"ERROR model should be {valid_models}")                
        return modelwrapper
    
    MODEL = my_model
    print(colored(f'\n### Loading model: {MODEL}', COLOR_DEBUG))
    
    tic = time.time()    
    
    if USE_FLEXGEN:
        modelwrapper_new = tm.flexgen_load_model(MODEL, USE_DEVICE)
    else:
        modelwrapper_new = tm.load_model(MODEL, USE_DEVICE)    
        
    toc = time.time()
    elapsed = toc - tic
    print(colored(f'### Elapsed: {elapsed:.2f}s', COLOR_DEBUG))
    
    return modelwrapper_new


def get_folder_size(folder: str) -> int:
    return sum(p.stat().st_size for p in Path(folder).rglob('*'))


def save_model_to_disk(modelwrapper):
    model = modelwrapper.model
    tokenizer = modelwrapper.tokenizer

    #metric = evaluate.load("accuracy")

    """training_args = TrainingArguments(
        output_dir="test_trainer_cpu_fp32",
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="epoch",
        per_device_train_batch_size=1,
        num_train_epochs=1,
        warmup_steps=WARMUP_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=DECAY,
        remove_unused_columns=True,
        use_ipex=True,
        no_cuda=True,
        fp16=False,
    )

    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

    trainer = Trainer(
        model=model,
        #args=training_args,
        train_dataset=[], #small_train_dataset_tk,
        #eval_dataset=small_eval_dataset_tk,
        #data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )


    #trainer.train()
"""

    trained_model_folder = f"{tm.TRAINED_FOLDER}/models/{modelwrapper.name}-full-trained"

    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False

    # save only DELTAs model
    print(f"### saving DELTA trained model {trained_model_folder}")
    model.save_pretrained(trained_model_folder)

    from peft import PeftModel
    if isinstance(model, PeftModel):
        print(f"### saving FULL trained model {trained_model_folder} ...")
        model.get_base_model().save_pretrained(trained_model_folder)

    # trainer.save_model(trained_model_folder)
    # trained_model_folder

    # save tokenizer and config
    tokenizer.save_pretrained(trained_model_folder)
    model.config.to_json_file(trained_model_folder + "/config.json")

    size = get_folder_size(trained_model_folder)

    model.config.use_cache = True
    print(
        f"### save_model_to_disk(): {trained_model_folder} size {size/1024/1024:.2f} Mbytes ")


def main():
    global USE_DEVICE, MAX_LENGTH, AUDIO_ENABLED, INITIAL_PROMPT, PROMPT_FOLDER, USE_FLEXGEN

    modelwrapper = switch_model(None, MODEL)

    seed = int(time.time())
    make_outpath()

    prompt = INITIAL_PROMPT
    # Top-P: Top-P is an alternative way of controlling the randomness of the generated text. We recommend that only one of Temperature and Top P are used, so when using one of them, make sure that the other is set to 1.
    # Top-K: sampling means sorting by probability and zero-ing out the
    # probabilities for anything below the k'th token. A lower value improves
    # quality by removing the tail and making it less likely to go off topic.

    opt = {
        "model_name": modelwrapper.name,
        "device": modelwrapper.use_device,
        "seed": seed,
        "prompt": "",
        "from_file": "",
        "list_from_file": "",
        "temperature": 0.4,
        "top_p": 1.0,
        "top_k": 50,
        "no_repeat_ngram_size": 3,
        "max_length": MAX_LENGTH,
        "max_time": 300.0,
        "num_return_sequences": 1
    }

    loop_cnt = 0
    chat_loop_cnt = 0
    chat_type = 0

    opt['prompt'] = prompt
    sep = "-" * 40

    while True:
        print(sep)
        print(colored(f"### Model {opt['model_name']} Device {opt['device']} Memory {modelwrapper.model.get_memory_footprint()/1024/1024:.2f}Mb CUDA Memory: {torch.cuda.memory_allocated()/1024/1024:.2f}Mb\n",COLOR_DEBUG))        
        print(colored(f"### Prompt: {prompt}", COLOR_PROMPT))                        
        skip_gen = False

        if loop_cnt > 0:
            print(f"### Looping generation {loop_cnt}")
            loop_cnt = loop_cnt - 1
        elif chat_loop_cnt > 0:
            print(f"### Looping chatbot generation {chat_loop_cnt}")
            chat_loop_cnt = chat_loop_cnt - 1
            if chat_loop_cnt == 0:
                skip_gen = True
        else:  
            print(colored(f"### Enter a valid command: {command_prompt.get_valid_commands()}", COLOR_PROMPT))
            command = command_prompt.command_prompt()
            if not command:
                continue
            if command[0] == 'help':
                print(colored("### Current options", COLOR_PROMPT))
                # print(f"### {model} {tokenizer}")                
                print(colored(f"### options: {opt}", COLOR_PROMPT))
                command_prompt.print_commands_help()
                continue

            if command[0] == 'quit':
                print("### quit\n")
                sys.exit(0)

            if command[0] == 'prompt':
                prompt = command[1]
                opt['prompt'] = prompt
                opt['from_file'] = ""
                opt['list_from_file'] = ""
                print("### prompt setting to " + prompt)
                skip_gen = True

            if command[0] == 'from_file':                
                from_file = PROMPT_FOLDER + "/" + command[1]
                opt['from_file'] = from_file
                opt['list_from_file'] = ""
                skip_gen = True

            if command[0] == 'list_from_file':    
                from_file = PROMPT_FOLDER + "/" + command[1]
                opt['list_from_file'] = from_file
                opt['from_file'] = ""
                skip_gen = True
                
            if command[0] == 'model':    
                new_model = command[1]
                modelwrapper = switch_model(modelwrapper, new_model)
                skip_gen = True

            if command[0] == 'seed':
                q = int(command[1])
                seed = q
                print(f"### Setting seed to {seed}")
                set_seed(seed)
                skip_gen = True

            if command[0] == 'loop':
                print("### loop\n")
                loop_cnt = int(command[1])
                skip_gen = False

            if command[0] == 'autochat':
                print("### Auto ChatBot\n")
                chat_loop_cnt = int(command[1])
                chat_type = 0
                skip_gen = False

            if command[0] == 'chat':
                print("### ChatBot\n")
                #chat_loop_cnt = int(command[1])
                chat_loop_cnt = 1 # testing, only 1 chat
                chat_type = 1
                skip_gen = False

            if command[0] == 'audio':
                AUDIO_ENABLED = int(command[1])
                print(f"### Audio setting {AUDIO_ENABLED}")
                skip_gen = True

            if command[0] == 'save_model':
                print(f"### save_model\n")
                save_model_to_disk(modelwrapper)
                skip_gen = True
                continue

            if command[0] == 'clean':
                global OUTPATH
                print(f"### clean output {OUTPATH}\n")
                clean_output_dir()
                skip_gen = True
                continue

            int_command = [
                'seed',
                'top_k',
                'no_repeat_ngram_size',
                'max_length',
                'max_time',
                'num_return_sequences']
            float_command = ['temperature', 'top_p']
            param_commands = int_command + float_command

            if command[0] in param_commands:
                key = command[0]
                value = command[1]
                print(f"### Setting {key} to {value}")
                opt[key] = value
                skip_gen = True

        if not skip_gen:
            set_seed(seed)
            l_prompts = []

            try:

                if chat_loop_cnt > 0:
                    print(f"### chatbot cnt {chat_loop_cnt}")
                    chat_bot(modelwrapper, opt, chat_type)
                elif len(opt['from_file']) > 0:
                    print("### reading prompt from file " + from_file)
                    x = read_prompt_from_file(from_file, parse_as_list=False)
                    if len(x) > 0:
                        prompt = x.strip()
                        l_prompts = [prompt]
                elif len(opt['list_from_file']) > 0:
                    print("### reading list of prompts from file " + from_file)
                    from_file = opt['list_from_file']
                    x = read_prompt_from_file(from_file, parse_as_list=True)
                    if len(x) > 0:
                        l_prompts = x
                else:
                    print("### using prompt " + opt['prompt'])
                    l_prompts = [opt['prompt']]

                if len(l_prompts) > 0:
                    generate_list_of_prompts(modelwrapper, l_prompts, opt)
                    seed = seed + 1
                    opt['seed'] = seed
                else:
                    print("### no prompt to generate")

            except KeyboardInterrupt:
                print('*** User abort')
                loop_cnt = 0
                chat_loop_cnt = 0
                pass
            except Exception as ex:
                print("!!! EXCEPTION " + str(ex))
                traceback.print_exc()
                loop_cnt = 0
                chat_loop_cnt = 0
                pass

# Namespace(model=None, device='CPU', prompt=None, testing=None, test_list_from_file=None)

def parse_command_line(opt):    
    global MODEL
    global USE_DEVICE
    
    global DO_TESTING
    global TEST_LIST_FROM_FILE
    global TEST_MODELS_LIST
    global INITIAL_PROMPT
    
    if opt.device:       
        USE_DEVICE=opt.device        
    if opt.model:
        MODEL=opt.model
    if opt.prompt:
        INITIAL_PROMPT=opt.prompt
        
    if opt.testing:
        DO_TESTING=True
    if opt.test_list_from_file:
        TEST_LIST_FROM_FILE=opt.test_list_from_file
    if opt.test_models:
        TEST_MODELS_LIST=opt.test_models.split(",")
        
    print(colored(f"### parse_command_line: {(USE_DEVICE,MODEL,INITIAL_PROMPT,DO_TESTING,TEST_LIST_FROM_FILE,TEST_MODELS_LIST)}",COLOR_DEBUG))
    

if __name__ == "__main__":
    opt = command_line.read_command_line()
    parse_command_line(opt)
    #sys.exit(0)
    
    if DO_TESTING:
        test_models()
    else:
        main()
