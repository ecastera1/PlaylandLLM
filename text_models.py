#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:35:30 2022
@author: Enrique Castera 
"""

import os
import time
import torch
from enum import Enum

from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import BertLMHeadModel, BertTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import MBartTokenizer, MBart50TokenizerFast, MBartForConditionalGeneration, BertForMaskedLM
from transformers import pipeline, set_seed
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelWithLMHead, AutoModelForSeq2SeqLM, GPTJForCausalLM, StoppingCriteria, StoppingCriteriaList
from transformers import BitsAndBytesConfig

from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig, prepare_model_for_int8_training

import accelerate


class DeviceClass(Enum):
    CUDA = 0,
    CUDA_FP16 = 1,
    CPU_INT8 = 2,
    CPU = 3,
    AUTO = 4


class ModelClass(Enum):
    GPTJ = 0
    GPT2 = 1
    BLOOM = 3
    ROBERTA = 4
    T5 = 5
    OPT = 6
    PYTHIA = 7
    ALPACA = 8
    FLAN = 9
    GPTNEOX = 10
    PEFT = 12


class ModelCard:
    def __init__(self, name, model_class, preferred_device=DeviceClass.CPU):
        self.name = name
        self.model_class = model_class
        self.preferred_device = preferred_device


class ModelWrapper:
    def __init__(
            self,
            name,
            model,
            tokenizer,
            generator,
            modelcard,
            use_device,
            use_flexgen=False):
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.generator = generator
        self.modelcard = modelcard
        self.use_device = use_device
        self.use_flexgen = use_flexgen


# GLOBAL VARIABLES
from text_models_config import *


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids: list):
        self.keywords = keywords_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False


def get_available_models():
    global MODELS_LIST
    return list(MODELS_LIST.keys())

def load_model_aux(my_model, use_device):
        
    if use_device == DeviceClass.CPU_INT8:
        print(f"### load_model_aux {my_model} in CPU INT8...")

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            load_in_8bit_fp32_cpu_offload = True,
            llm_int8_enable_fp32_cpu_offload=True)

        #custom_device_map = {'':'cpu'}
        #custom_device_map = build_device_map()

        model = AutoModelForCausalLM.from_pretrained(
            my_model,
            load_in_8bit=True,            
            low_cpu_mem_usage=True,

            device_map='auto',
            # device_map=custom_device_map,
            # torch_dtype=torch.float32,

            # device_map='auto',
            torch_dtype=torch.float16,

            quantization_config=quantization_config,
            #max_memory={0: '11GB', 'cpu': '32GB'},
            #max_memory={0: '7GB', 'cpu': '32GB'},
            offload_state_dict=True,
            offload_folder="./offload"
        )

        print(f"### Device map {model.hf_device_map}")

    elif use_device == DeviceClass.CPU:
        print(f"### load_model_aux {my_model} in CPU FP32...")

        model = AutoModelForCausalLM.from_pretrained(
            my_model,
            load_in_8bit=False,
            low_cpu_mem_usage=True,

            # device_map=custom_device_map,
            # torch_dtype=torch.float32,

            # device_map='auto',
            # torch_dtype=torch.float16,

            # quantization_config=quantization_config,
            #max_memory={0: '11GB', 'cpu': '32GB'},
            offload_state_dict=True,
            offload_folder="./offload"
        )
        device = torch.device("cpu")
        model.to(device)

    elif use_device == DeviceClass.CUDA_FP16:
        print(f"### load_model_aux {my_model} in CUDA FP16 Half...")

        # quantization_config = BitsAndBytesConfig(
        #        load_in_8bit = True,
        #        llm_int8_enable_fp32_cpu_offload=True)

        custom_device_map = {'': 'cuda:0'}

        model = AutoModelForCausalLM.from_pretrained(
            my_model,
            load_in_8bit=False,
            low_cpu_mem_usage=True,

            # device_map='auto',
            # device_map=custom_device_map,
            torch_dtype=torch.float16,

            # quantization_config=quantization_config,
            #max_memory={0: '11GB', 'cpu': '32GB'},
            offload_state_dict=True,
            offload_folder="./offload"
        )

        # print(f"### Device map {model.hf_device_map}")
    elif use_device == DeviceClass.CUDA:
        print(f"### load_model_aux {my_model} in CUDA ...")

        # quantization_config = BitsAndBytesConfig(
        #        load_in_8bit = True,
        #        llm_int8_enable_fp32_cpu_offload=True)

        custom_device_map = {'': 'cuda:0'}

        model = AutoModelForCausalLM.from_pretrained(
            my_model,
            load_in_8bit=False,
            low_cpu_mem_usage=True,

            device_map='auto',
            # device_map=custom_device_map,
            # torch_dtype=torch.float16,

            # quantization_config=quantization_config,
            #max_memory={0: '11GB', 'cpu': '32GB'},
            offload_state_dict=True,
            offload_folder="./offload"
        )

        print(f"### Device map {model.hf_device_map}")
    else:
        print(f"### ERROR Device not supported {use_device} !!!")
        return None

    print(f"### load_model_aux complete {my_model} {model.device}")
    print(f"### Memory {model.get_memory_footprint()/1024/1024:.2f}Mb CUDA Memory: {torch.cuda.memory_allocated()/1024/1024:.2f}Mb")

    return model


def load_model(model_id, use_device):
    global MODELS_LIST, my_stopping_criteria
    modelcard = MODELS_LIST[model_id]
    use_model = modelcard.name

    print(f"### Loading {modelcard.model_class} model: {use_model} from .cache ...")

    models_causallm = (ModelClass.GPTJ,
                       ModelClass.BLOOM,
                       ModelClass.OPT,
                       ModelClass.ROBERTA,
                       ModelClass.GPT2,
                       ModelClass.ALPACA
                       )

    if use_device == DeviceClass.AUTO:
        my_device = modelcard.preferred_device
    else:
        my_device = use_device
        
    if torch.cuda.is_available() == False:
        if my_device == DeviceClass.CUDA_FP16 or my_device == DeviceClass.CUDA:
            print(f"### ERROR trying to load_model {use_model} in {my_device} but CUDA is not available. Forcing to CPU...")
            my_device = DeviceClass.CPU
       
    

    if modelcard.model_class in models_causallm:
        tokenizer = AutoTokenizer.from_pretrained(use_model)
        # with accelerate.init_empty_weights():
        model = load_model_aux(use_model, my_device)
        # model.tie_weights()

    if modelcard.model_class == ModelClass.GPTNEOX:

        tokenizer = AutoTokenizer.from_pretrained(use_model)
        model = load_model_aux(use_model, my_device)

        """quantization_config = BitsAndBytesConfig(
             load_in_8bit = True,
             llm_int8_enable_fp32_cpu_offload=True)

         custom_device_map = {
            'gpt_neox.embed_in': 'cuda:0',
            #'gpt_neox.layers': 'cpu',
            'gpt_neox.final_layer_norm': 'cpu',
            'embed_out': 'cpu'
            }

         # tiene layers de 1 a 43
         # metemos las primeras en cuda y el resto en CPU
         # hasta llenar la memoria de la GPU monitorizarlo con gpu_top.sh
         # mas o menos 15 layers
         LAYERS_IN_CUDA = 18
         for x in range(44):
             if x>LAYERS_IN_CUDA:
                 d="cpu"
             else:
                 d="cuda:0"
             key="gpt_neox.layers."+str(x)
             custom_device_map[key]=d

         model = AutoModelForCausalLM.from_pretrained(
                use_model,
                #revision="float16"

                low_cpu_mem_usage = modelcard.low_cpu_mem_usage,
                load_in_8bit=True,
                device_map = custom_device_map,
                #device_map = "auto",

                quantization_config=quantization_config,

                torch_dtype=torch.float16,
                max_memory={'cuda:0': '11GB', 'cpu': '32GB'},
                offload_state_dict=True,
                offload_folder="./offload"
                )
         """

    if modelcard.model_class == ModelClass.PEFT:
        trained_model_folder = TRAINED_FOLDER + "/models/" + use_model
        #trained_model_folder = "./models/" + use_model
        
        print(f"### Loading PEFT model: {use_model} from {trained_model_folder} ...")

        config = PeftConfig.from_pretrained(
            pretrained_model_name_or_path=trained_model_folder)
        model = load_model_aux(config.base_model_name_or_path, my_device)
        model = PeftModel.from_pretrained(model, trained_model_folder)
        tokenizer = AutoTokenizer.from_pretrained(trained_model_folder)

        #use_model = use_model +"-trained"

    if modelcard.model_class == ModelClass.PYTHIA:
        print(f"### Loading PYTHIA model: {use_model} from .cache ...")
        tokenizer = AutoTokenizer.from_pretrained(use_model)
        model = load_model_aux(use_model, my_device)

        tokenizer.pad_token = tokenizer.eos_token
        stop_token = "<|stop|>"
        tokenizer.add_tokens([stop_token])
        stop_ids = [tokenizer.encode(w)[0] for w in [stop_token]]
        stop_criteria = KeywordsStoppingCriteria(stop_ids)
        my_stopping_criteria = StoppingCriteriaList([stop_criteria])

        """if modelcard.float16==True:
            model = AutoModelForCausalLM.from_pretrained(
                use_model,
                #revision="float16"
                torch_dtype=torch.float16,
                low_cpu_mem_usage = modelcard.low_cpu_mem_usage,
                load_in_8bit=True,
                device_map="auto"
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(use_model, low_cpu_mem_usage = modelcard.low_cpu_mem_usage)
        """

    if modelcard.model_class == ModelClass.T5:
        print(f"Loading T5 model: {use_model} from .cache ...")
        tokenizer = AutoTokenizer.from_pretrained(
            use_model, padding_side='left')
        #model = load_model_aux(use_model, my_device)

        if my_device == DeviceClass.CPU_INT8:

            tokenizer = AutoTokenizer.from_pretrained(use_model)
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True)

            custom_device_map = {
                'gpt_neox.embed_in': 'cuda:0',
                # 'gpt_neox.layers': 'cpu',
                'gpt_neox.final_layer_norm': 'cpu',
                'embed_out': 'cpu'
            }

            # FIXIT: tiene layers de 1 a 43
            # metemos las primeras en cuda y el resto en CPU
            # hasta llenar la memoria de la GPU monitorizarlo con gpu_top.sh
            # mas o menos 15 layers
            LAYERS_IN_CUDA = 18
            for x in range(44):
                if x > LAYERS_IN_CUDA:
                    d = "cpu"
                else:
                    d = "cuda:0"
                key = "gpt_neox.layers." + str(x)
                custom_device_map[key] = d

            model = AutoModelWithLMHead.from_pretrained(
                use_model,
                # revision="float16"
                low_cpu_mem_usage=True,
                load_in_8bit=True,
                device_map="auto",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                #max_memory={'cuda:0': '11GB', 'cpu': '32GB'},
                offload_state_dict=True,
                offload_folder="./offload"
            )
        else:
            model = AutoModelWithLMHead.from_pretrained(use_model)
            #model = AutoModelForSeq2SeqLM.from_pretrained(use_model, low_cpu_mem_usage = modelcard.low_cpu_mem_usage)

    if modelcard.model_class == ModelClass.FLAN:
        print(f"### Loading FLAN model: {use_model} from .cache ...")
        tokenizer = AutoTokenizer.from_pretrained(
            use_model, padding_side='left')
        model = AutoModelForSeq2SeqLM.from_pretrained(
            use_model, low_cpu_mem_usage=modelcard.low_cpu_mem_usage)

    """if modelcard.model_class == ModelClass.ALPACA:
        print(f"### Loading ALPACA model: {use_model} from .cache ...")

        use_half = False
        if use_half:
            def noop(*args, **kwargs):
                pass

            torch.nn.init.kaiming_uniform_ = noop
            torch.nn.init.uniform_ = noop
            torch.nn.init.normal_ = noop

            torch.set_default_dtype(torch.half)
            from transformers import modeling_utils
            modeling_utils._init_weights = False
            torch.set_default_dtype(torch.half)

        #tokenizer = LlamaTokenizer.from_pretrained(use_model, padding_side='left')
        #model = LlamaForCausalLM.from_pretrained(use_model, low_cpu_mem_usage = modelcard.low_cpu_mem_usage)
        tokenizer = AutoTokenizer.from_pretrained(use_model, padding_side='left')
        #model = AutoModelForSeq2SeqLM.from_pretrained(use_model, low_cpu_mem_usage = modelcard.low_cpu_mem_usage)
        model = AutoModelForCausalLM.from_pretrained(use_model,low_cpu_mem_usage = modelcard.low_cpu_mem_usage)
    """

    #print("Model Parameters:")
    # p_dict=dict(model.named_parameters())
    # for k in p_dict.keys():
    #    #print(type(param), param.size())
    #    print(str(k))

    print(f"### Tokenizer model_max_length = {tokenizer.model_max_length}")
    print(f"### Tokenizer max_model_input_sizes = {tokenizer.max_model_input_sizes}")

    if my_device == DeviceClass.CUDA:
        print("### DEVICE: Model can run in GPU forcing to cuda:0")
        USE_DEVICE = "cuda:0"

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # print(torch.cuda.memory_allocated())

    if my_device == DeviceClass.CUDA_FP16:
        print("### DEVICE: Model CUDA_FP16")
        model.half()
        USE_DEVICE = "cuda:0"

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # print(torch.cuda.memory_allocated())

    if my_device == DeviceClass.CPU:
        print("### DEVICE: Model can only run in CPU")
        USE_DEVICE = "cpu"

    if my_device == DeviceClass.CPU_INT8:
        print("### DEVICE: Model in CPU_INT8")
        USE_DEVICE = "auto"

    if USE_DEVICE != "auto":
        device = torch.device(USE_DEVICE)
        print("### DEVICE: Using torch device ", device)
        model.to(device)
        generator = pipeline('text-generation',
                             model=model,
                             tokenizer=tokenizer,
                             device=USE_DEVICE)
    else:
        print("### DEVICE: Using torch device AUTO, not forcing device in transformers")
        generator = pipeline('text-generation',
                             model=model,
                             tokenizer=tokenizer
                             )

    model.eval()
    print(f"### Loaded complete {use_model} on device {USE_DEVICE}")
    print(f"### Memory {model.get_memory_footprint()/1024/1024:.2f}Mb CUDA Memory: {torch.cuda.memory_allocated()/1024/1024:.2f}Mb")

    return ModelWrapper(
        use_model,
        model,
        tokenizer,
        generator,
        modelcard,
        USE_DEVICE,
        use_flexgen=False)


"""
#
# FLEXGEN support, needs more testing
#

from flexgen.flex_opt import (
    Policy,
    OptLM,
    TorchDevice,
    TorchDisk,
    TorchMixedDevice,
    CompressionConfig,
    Env,
    Task,
    get_opt_config,
    str2bool)

def flexgen_load_model(model_id, use_device):
    global MODELS_LIST, my_stopping_criteria

    percent = (100, 0, 100, 0, 100, 0)
    # Six numbers. They are
    # "the percentage of weight on GPU, "
    # "the percentage of weight on CPU, "
    # "the percentage of attention cache on GPU, "
    # "the percentage of attention cache on CPU, "
    # "the percentage of activations on GPU, "
    # "the percentage of activations on CPU")
    offload_dir = "."  # "~/flexgen_offload_dir",
    path = "."

    # Initialize environment
    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(offload_dir)
    env = Env(gpu=gpu, cpu=cpu, disk=disk,
              mixed=TorchMixedDevice([gpu, cpu, disk]))

    print(f" device mix {[gpu,cpu,disk]} args {percent}")

    # Offloading policy
    policy = Policy(1, 1,
                    percent[0], percent[1],
                    percent[2], percent[3],
                    percent[4], percent[5],
                    overlap=True, sep_layer=True, pin_weight=False,
                    cpu_cache_compute=False, attn_sparsity=1.0,
                    compress_weight=True,
                    comp_weight_config=CompressionConfig(
                        num_bits=4, group_size=64,  # num_bits=8 group_size=64
                        group_dim=0, symmetric=False),
                    compress_cache=True,
                    comp_cache_config=CompressionConfig(
                        num_bits=4, group_size=64,
                        group_dim=2, symmetric=False))

    # Model
    modelcard = MODELS_LIST[model_id]
    use_model = modelcard.name

    print(f"flexgen_load_model {use_model}")

    tokenizer = AutoTokenizer.from_pretrained(use_model, padding_side="left")
    tokenizer.add_bos_token = True

    #stop_ids = tokenizer("\n").input_ids[0]
    #stop_criteria = KeywordsStoppingCriteria(stop_ids)
    #my_stopping_criteria = StoppingCriteriaList([stop_criteria])

    #stop = tokenizer(".").input_ids[0]
    #stop = tokenizer(HUMAN_NAME).input_ids[0]

    tic = time.time()
    print("Initialize...")

    opt_config = get_opt_config(use_model)
    model = OptLM(opt_config, env, path, policy)
    model.init_all_weights()

    print("Initialize done.")
    toc = time.time()
    elapsed = toc - tic
    print(f'### Elapsed: {elapsed:.2f}s')

    print(f"Tokenizer model_max_length = {tokenizer.model_max_length}")
    print(
        f"Tokenizer max_model_input_sizes = {tokenizer.max_model_input_sizes}")

    return ModelWrapper(
        use_model,
        model,
        tokenizer,
        None,
        modelcard,
        "flexgen",
        use_flexgen=True)


def flexgen_generate(modelwrapper, prompt, temperature=0.5, max_length=100):
    stop_ids = modelwrapper.tokenizer("\n").input_ids[0]

    inputs = modelwrapper.tokenizer([prompt])

    output_ids = modelwrapper.model.generate(
        inputs.input_ids,
        do_sample=True,
        temperature=temperature,
        max_new_tokens=max_length,
        stop=stop_ids)

    outputs = modelwrapper.tokenizer.batch_decode(
        output_ids, skip_special_tokens=True)[0]

    response_without_prompt = outputs[len(prompt):].strip()

    print(f"flexgen_generate {response_without_prompt}")
    return response_without_prompt
"""
