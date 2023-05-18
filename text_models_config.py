#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 14:25:39 2023

Global Configuration for text_models.py module

@author: quique
"""
from text_models import ModelCard, ModelClass, DeviceClass

TRAINED_FOLDER = "/home/quique/text-models/training"


MODELS_LIST_ES = {
    "gpt2-spanish": ModelCard("mrm8488/spanish-gpt2", ModelClass.GPT2, preferred_device=DeviceClass.CUDA),
    "gpt2-small-spanish": ModelCard("datificate/gpt2-small-spanish", ModelClass.GPT2, preferred_device=DeviceClass.CUDA),

    "gpt2-flax-spanish": ModelCard("flax-community/gpt-2-spanish", ModelClass.GPT2, preferred_device=DeviceClass.CUDA),
    "gpt2-deepesp-spanish": ModelCard("DeepESP/gpt2-spanish", ModelClass.GPT2, preferred_device=DeviceClass.CUDA),

    "bertin-gpt-j-6b-half-sharded": ModelCard("DavidFM43/bertin-gpt-j-6b-half-sharded", ModelClass.GPTJ, preferred_device=DeviceClass.CPU_INT8),
    "gptj-bertin-alpaca": ModelCard("bertin-project/bertin-gpt-j-6B-alpaca", ModelClass.GPTJ, preferred_device=DeviceClass.CPU_INT8),
    "gptj-bertin-libros": ModelCard("bertin-project/bertin-gpt-j-6B-infolibros", ModelClass.GPTJ, preferred_device=DeviceClass.CPU_INT8),
    "gptj-bertin": ModelCard("bertin-project/bertin-gpt-j-6B", ModelClass.GPTJ, preferred_device=DeviceClass.CPU_INT8)
}

MODELS_LIST_EN = {
    "gpt2": ModelCard("gpt2", ModelClass.GPT2, preferred_device=DeviceClass.CUDA),

    "bloom-7b1": ModelCard("bigscience/bloom-7b1", ModelClass.BLOOM, preferred_device=DeviceClass.CPU_INT8),
    "bloomz-3b": ModelCard("bigscience/bloomz-3b", ModelClass.BLOOM, preferred_device=DeviceClass.CUDA_FP16),
    "bloom-3b": ModelCard("bigscience/bloom-3b", ModelClass.BLOOM, preferred_device=DeviceClass.CUDA_FP16),
    "bloom-sd-prompts": ModelCard("mrm8488/bloom-560m-finetuned-sd-prompts", ModelClass.BLOOM, preferred_device=DeviceClass.CUDA),

    "opt-30b": ModelCard("facebook/opt-30b", ModelClass.OPT, preferred_device=DeviceClass.CPU),
    "opt-6.7b": ModelCard("facebook/opt-6.7b", ModelClass.OPT, preferred_device=DeviceClass.CPU_INT8),
    "opt-1.3b": ModelCard("facebook/opt-1.3b", ModelClass.OPT, preferred_device=DeviceClass.CUDA),

    "gptj": ModelCard("EleutherAI/gpt-j-6B", ModelClass.GPTJ, preferred_device=DeviceClass.CPU_INT8),
    "gptjt-together": ModelCard("togethercomputer/GPT-JT-6B-v1", ModelClass.GPTJ, preferred_device=DeviceClass.CPU_INT8),
    "gpt-j-6B-alpaca-gpt4": ModelCard("vicgalle/gpt-j-6B-alpaca-gpt4", ModelClass.GPTJ, preferred_device=DeviceClass.CPU_INT8),

    "pythia-1.4b": ModelCard("lambdalabs/pythia-1.4b-deduped-synthetic-instruct", ModelClass.PYTHIA, preferred_device=DeviceClass.CUDA),
    "pythia-2.8b": ModelCard("lambdalabs/pythia-2.8b-deduped-synthetic-instruct", ModelClass.PYTHIA, preferred_device=DeviceClass.CUDA_FP16),
    "pythia-3b-sft": ModelCard("theblackcat102/pythia-3b-deduped-sft-r1", ModelClass.PYTHIA, preferred_device=DeviceClass.CUDA_FP16),
    "pythia-6.9b": ModelCard("lambdalabs/pythia-6.9b-deduped-synthetic-instruct", ModelClass.PYTHIA, preferred_device=DeviceClass.CPU_INT8),

    "flan-alpaca-220m-base": ModelCard("declare-lab/flan-alpaca-base", ModelClass.T5, preferred_device=DeviceClass.CUDA),
    "flan-alpaca-gpt4-xl": ModelCard("declare-lab/flan-alpaca-gpt4-xl", ModelClass.T5, preferred_device=DeviceClass.CPU_INT8),

    # "flan-t5-xxl": ModelCard("google/flan-t5-xxl", ModelClass.T5, float16=False, low_cpu_mem_usage=False, can_run_gpu=False),
    # "flan-t5-xl": ModelCard("google/flan-t5-xl", ModelClass.T5, float16=False, low_cpu_mem_usage=True, can_run_gpu=True),
    # "flan-t5-large": ModelCard("google/flan-t5-large", ModelClass.T5, float16=False, low_cpu_mem_usage=True, can_run_gpu=True),
    "WizardLM-7B-Uncensored": ModelCard("ehartford/WizardLM-7B-Uncensored", ModelClass.ALPACA, preferred_device=DeviceClass.CPU_INT8),
    "Vicuna-EvolInstruct-7B": ModelCard("LLMs/Vicuna-EvolInstruct-7B", ModelClass.ALPACA, preferred_device=DeviceClass.CPU_INT8),


    "GPT-NeoXT-Chat-Base-20B": ModelCard("togethercomputer/GPT-NeoXT-Chat-Base-20B", ModelClass.GPTNEOX, preferred_device=DeviceClass.CPU),
    "gpt-neox-20b": ModelCard("EleutherAI/gpt-neox-20b", ModelClass.GPTNEOX, preferred_device=DeviceClass.CPU)

}

MODEL_LIST_TRAINED = {
    "gpt2-small-spanish-telegram-trained": ModelCard("datificate/gpt2-small-spanish-telegram-trained", ModelClass.PEFT, preferred_device=DeviceClass.CPU),
    "gpt2-small-spanish-telegram-trained-full-trained": ModelCard(TRAINED_FOLDER + "/models/datificate/gpt2-small-spanish-telegram-trained-full-trained", ModelClass.GPT2, preferred_device=DeviceClass.CPU),
    "gpt2-small-spanish-alpaca-es-trained": ModelCard(TRAINED_FOLDER + "/models/datificate/gpt2-small-spanish-alpaca-es-trained", ModelClass.GPT2, preferred_device=DeviceClass.CUDA),


    "gpt2-small-spanish-marte_trilogia-trained": ModelCard("datificate/gpt2-small-spanish-marte_trilogia-trained", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8),

    "bloomz-3b-burbuja-trained": ModelCard("bigscience/bloomz-3b-burbuja-trained", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8),

    "bertin-gpt-j-6B-wiki_fisica-trained": ModelCard("bertin-project/bertin-gpt-j-6B-wiki_fisica-trained", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8),
    "bertin-gpt-j-6B-marte_sentences-trained": ModelCard("bertin-project/bertin-gpt-j-6B-marte_sentences-trained", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8),
    "bertin-gpt-j-6B-ringworld-trained": ModelCard("bertin-project/bertin-gpt-j-6B-ringworld-trained", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8),
    "bertin-gpt-j-6B-telegram-trained": ModelCard("bertin-project/bertin-gpt-j-6B-telegram-trained", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8),
    "bertin-gpt-j-6B-burbuja-trained": ModelCard("bertin-project/bertin-gpt-j-6B-burbuja-trained", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8),

    "Vicuna-EvolInstruct-7B-alpaca-es-trained": ModelCard("LLMs/Vicuna-EvolInstruct-7B-alpaca-es-trained", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8),
    "Vicuna-EvolInstruct-7B-alpaca-es-trained-full-trained": ModelCard(TRAINED_FOLDER + "/models/LLMs/Vicuna-EvolInstruct-7B-alpaca-es-trained-full-trained", ModelClass.ALPACA, preferred_device=DeviceClass.CPU_INT8),


    "gpt2-trained": ModelCard("gpt2", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8),
    "gptj-trained": ModelCard("EleutherAI/gpt-j-6B", ModelClass.PEFT, preferred_device=DeviceClass.CPU_INT8)
}

MODELS_LIST_NOT_WORKING = {
    # "gpt2-biblioteca-nacional-large": ModelCard("PlanTL-GOB-ES/gpt2-large-bne", ModelClass.GPT2, float16=False, low_cpu_mem_usage=False, can_run_gpu=True),
    # "gpt2-biblioteca-nacional": ModelCard("PlanTL-GOB-ES/gpt2-base-bne", ModelClass.GPT2, float16=False, low_cpu_mem_usage=False, can_run_gpu=True),
    # "roberta-large-bne": ModelCard("PlanTL-GOB-ES/roberta-large-bne", ModelClass.ROBERTA, float16=False, low_cpu_mem_usage=False, can_run_gpu=True),
    # "bloom-1b1-spanish": ModelCard("jorgeortizfuentes/bloom-1b1-spanish", ModelClass.BLOOM, float16=False, low_cpu_mem_usage=True, can_run_gpu=False)

    # "gpt-j-6B-8bit": ModelCard("TianXxx/gpt-j-6b-8bit", ModelClass.GPTJ, float16=False, low_cpu_mem_usage=True, can_run_gpu=False),
    # "t5-e2m-intent": ModelCard("mrm8488/t5-base-finetuned-e2m-intent", ModelClass.T5, float16=False, low_cpu_mem_usage=True, can_run_gpu=True),
    # "facebook/opt-66b": ModelCard("facebook/opt-66b", ModelClass.OPT, float16=False, low_cpu_mem_usage=True, can_run_gpu=False),
    # "alpaca-native-4bit": ModelCard("ozcur/alpaca-native-4bit", ModelClass.ALPACA, float16=False, low_cpu_mem_usage=True, can_run_gpu=True)
    # "flan-alpaca-3b-xl": ModelCard("declare-lab/flan-alpaca-xl", ModelClass.T5, float16=False, low_cpu_mem_usage=True, can_run_gpu=True),
    # "flan-sharegpt-xl": ModelCard("declare-lab/flan-sharegpt-xl", ModelClass.T5, float16=False, low_cpu_mem_usage=True, can_run_gpu=True),
    # "stablelm-base-alpha-3b": ModelCard("stabilityai/stablelm-base-alpha-3b", ModelClass.T5, float16=False, low_cpu_mem_usage=True, can_run_gpu=False),

}

MODELS_LIST = MODELS_LIST_ES | MODELS_LIST_EN | MODEL_LIST_TRAINED
