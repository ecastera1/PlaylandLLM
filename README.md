# PlaylandLLM

A python app with CLI interface to do local inference and testing of open source LLMs for text-generation. Test any transformer LLM community model such as GPT-J, Pythia, Bloom, LLaMA, Vicuna, Alpaca, or any other model supported by Huggingface's transformer text-generation interface. Run model locally in your computer without the need of 3rd party paid APIs or keys.

# Summary

PlaylandLLM is a set of python tools to simplify inference testing of text-generation LLM models locally in your computer without the need of 3rd party paid APIs

* Encapsulate Pytorch and Transformers complexities of dealing with large models with limited resources
* Loading and inference for GPU, CPU, datatypes FP32, FP16, INT8, quantization, spread accross devices, etc.
* Support for **PEFT adapters** models
* Support for **bitsandbytes, accelerate**
* Test any **Huggingface** models from the community from different families **GPT-J, Pythia, Bloom, LLaMA, Vicuna, Alpaca**, or any other LLM supported by Huggingface's transformer text-generation locally in your computer without the need of 3rd party paid APIs or keys.
* Compare model performance and outputs.
* Command line readline interface CLI to interactively enter prompts, change runtime model paramenters, chat from terminal. History, tab completion... Easy to add more commands.
* Loading time is critical in testing as model sizes increases. Load the model once, interact with it while keeping in memory. 
* Logging subsystem for model outputs, chat outputs, etc. it will save responses locally in txt files that can easily check it later
* Testing framework to test several models & prompts offline
* Interactive chat mode with simple memory for chatbot-like experience
* Auto chat mode to generate an auto-conversation
* Use your own prompt database and throw it at the models you want to test
* Audio TTS Output in **Spanish or English** thanks to ```snakers4/silero-models```

### Credits
* Using Transformers API and Huggingface.co classes (https://huggingface.co/) and pytorch
* bitsandbytes INT8 optimizations (https://github.com/TimDettmers/bitsandbytes)
* Audio Text-to-Speech TTS silero models (https://github.com/snakers4/silero-models/blob/master/README.md)
* Audio Playback SoX https://sox.sourceforge.net/


## Installing
```
pip install -r requirements.txt
```
### Requirements
* torch==2.0.0
* transformers==4.28.1
* peft==0.3.0
* accelerate==0.18.0
* bitsandbytes==0.37.2
> Note that there's a bug in 0.38.0 you must use 0.37.2 to avoid out of memory errors (https://github.com/TimDettmers/bitsandbytes/issues/324)


## Usage
I haven't implemented yet full command line parsing since I'm focusing on experiments and automation within code. Code is easy to read and all the important variables are global in capital. As minimum setup you need to edit a few parameters controlling the basics:

* Edit playlandllm.py as you see fit
* Specify model and device to use
* Fine tune other parameters such as language, audio output (TTS), max lengths...
* Launch in interactive mode single model interactivity with inference (default mode) or
* Launch in Testing mode, run tests on a list of models

## Command line

```
python playlandllm.py
```


## Device and accelerator support used for inference

Possible values are 
* AUTO - Default. It will use modelcard's model configuration as defined in text_models.py module
* CUDA - force Nvidia cuda device if you have GPU capable device. This is the fastest if you have enough GPU VRAM to fit the model parameters
* CPU_INT8 - use bitsandbytes and accelerate, INT8 quantization and offloading weights, transformers load_in_8bits and other optimizations trying to fit large models in limited resources. This will use CUDA devices as well spreading your modules and parameters accross devices.
* CPU - use CPU and FP32 fitting everything in RAM, slow

`text_models.py` have all the templates for lots of HuggingFace models that I've been experimenting with.

Device selection is controlled by this variable

```
import text_models as tm
USE_DEVICE = tm.DeviceClass.AUTO 
```



## System Setup and expected performance
My system setup is:
* 32Gb RAM + 64Gb swap, 1 Tbyte SSD
* 11Gb NVIDIA GeForce RTX 3060
* CPU: Intel(R) Core(TM) i5-10600K CPU @ 4.10GHz
* Ubuntu 22.04.2 LTS

With this I can run inference locally with good performance, especially when using INT8 quantization techniques. For your reference these are the results I'm getting:

* 13 Billion models and larger GPTX-Neox family - these I can run in CPU only, very slow in my system but I can test it. EleutherAI/gpt-neox-20b avg response time is 150-300 secs.
* 6 Billion models - GPT-J family, Pythia-6B, opt-7B, Vicuna-7B can all run with CPU_INT8 optimizations. Takes time to load model but then inference is <30 secs per prompt.
* 3 Billion models - these can fit happily in CUDA 11Gb in FP16. Inference is <12 secs. This is the sweet spot for my system, good compromise between quality and performance. Highly usable in interactive mode for chatbots, etc.
* <1 Billion models GPT-2-like and smaller models, inference is <2 secs.


## Model selection
* A list of models is pre-configured in ```text_models.py```, basically a dictionary with model default parameters. Set MODEL to one of these keys to select model to use for inference.
* You must edit ```text_models.py``` to add the models that you want to test.

```
MODEL="gptj-bertin-alpaca"
```

## Testing models

* In this mode instead of running interactive command-line mode you can test several models sequentially feeding those with a list of prompts. 
* Leave it running offline, system will be logging all outputs and stats.
* Check and compare outputs and performance. It will save stats and KPIs so you can check real-world inference performance performance and quality of the outputs on your system.

1. Set ```DO_TESTING``` variable to True

```
DO_TESTING = True
```

2. Set a text file containing the list of prompts to feed into the models

```
TEST_LIST_FROM_FILE = "logic_prompts.txt"
```

3. Define the list of models to test in ```TEST_MODELS_LIST```:

```
TML_COMPLETE_ES = (
    "gpt2-spanish",
    "gpt2-small-spanish-trained",

    "gpt2-small-spanish",
    "gpt2-deepesp-spanish",
    "gpt2-biblioteca-nacional",
    "gpt2-biblioteca-nacional-large",
    "gpt2-flax-spanish",

    "gptj-bertin",
    "gptj-bertin-libros",
    "gptj-bertin-alpaca",
    "bertin-gpt-j-6b-half-sharded"
)

TEST_MODELS_LIST = TML_COMPLETE_ES
```

4. Launch and leave it running.

```
python playlandllm.py
```

5. While executing, it will create a folder structure with one folder per model and all prompt tests results as txt files with key info: prompt, output and runtime parameters for your later examination. Example output for a single prompt:


```
--------------------
### output 1
Escribe una pequeña historia sobre un personaje que acaba de descubrir un talento oculto.

### Respuesta:
Juan siempre había sido una persona corriente, sin ningún talento especial o habilidad destacable. Pero un día, mientras se duchaba, descubrió una nueva pasión. Comenzó a practicar en la ducha y pronto su habilidades comenzaron a fluir. Cada vez se sentía más seguro y práctico en la bañera y en la calle. La gente empezó a notar su nuevo talento y prontó comenzó un negocio de baile bajo el agua. Con el tiempo, su talento secreto se difundió como un reguero de pólvora y prósperó. Juan era ahora una figura destacada en el mundo del espectáculo y estaba viviendo la vida de sus sueños.
--------------------
opt = {'model_name': 'bertin-project/bertin-gpt-j-6B-alpaca', 'device': 'auto', 'seed': 1682338949, 'prompt': 'Escribe una pequeña historia sobre un personaje que acaba de descubrir un talento oculto', 'from_file': '', 'list_from_file': 'es_lprompts1.txt', 'temperature': 0.4, 'top_p': 1.0, 'top_k': 50, 'no_repeat_ngram_size': 3, 'max_length': 256, 'max_time': 300.0, 'num_return_sequences': 1}
prompt = Escribe una pequeña historia sobre un personaje que acaba de descubrir un talento oculto
elapsed time = 23.37s
```

```
--------------------
### output 1
Write a poem about shiba inu pet and his master.
Answer:Slender is the shiba Inu pet, his master's shadowed side. 
He prowls through the night, his body agile 
His quick mind filled with tricks, his gentle heart 
A perfect match for his master, his owner.

His fur is a mix of black and white, 
 His eyes are a curious blue, his tail a swirly wag. 
		 He's a loyal companion, always by his side.
His gentle nature makes him an ideal pet.

			2 
		 Slender is also strong, 
		 A force to be reckoned with, 
			 His quick mind and sharp claws.
		
			3 
		 His body is short and stout, 
				 His coat is soft and thick, 
					 His gentle heart is pure.
		
			4 
		 No matter the weather, 		
		 His master is always there, 
						 His loyal companion.
		
				5 
		 Master, you are his home, 			
			 His world is your backyard, 				
			 His perfect match, his beloved pet.<|stop|>.se/poems/shiba-inu
--------------------
opt = {'model_name': 'lambdalabs/pythia-2.8b-deduped-synthetic-instruct', 'device': 'cuda:0', 'seed': 1682330282, 'prompt': 'Write a poem about shiba inu pet and his master.', 'from_file': '', 'list_from_file': 'lprompts1.txt', 'temperature': 0.4, 'top_p': 1.0, 'top_k': 50, 'no_repeat_ngram_size': 3, 'max_length': 256, 'max_time': 300.0, 'num_return_sequences': 1}
prompt = Write a poem about shiba inu pet and his master.
elapsed time = 11.16s
```


6. It will also create results.csv with a summary of performance stats for each test. This is what I got on my system for a set of models.


id|model|device|total_time|avg_chars_per_sec|avg_response_length|
---|---|---|---|---|---
0|gpt2|cuda:0|29.69|96.73|24.75
1|declare-lab/flan-alpaca-gpt4-xl|auto|364.24|7.85|24.05
2|declare-lab/flan-alpaca-base|cuda:0|20.59|135.79|20.95
3|lambdalabs/pythia-1.4b-deduped-synthetic-instruct|cuda:0|137.53|25.76|58.30
4|lambdalabs/pythia-2.8b-deduped-synthetic-instruct|cuda:0|153.42|22.32|52.40
5|lambdalabs/pythia-6.9b-deduped-synthetic-instruct|auto|431.72|7.19|36.35
6|theblackcat102/pythia-3b-deduped-sft-r1|cuda:0|149.70|23.41|56.40
7|EleutherAI/gpt-j-6B|auto|536.59|6.74|61.95
8|togethercomputer/GPT-JT-6B-v1|auto|571.76|5.97|51.85
9|vicgalle/gpt-j-6B-alpaca-gpt4|auto|509.70|6.75|53.20
10|bigscience/bloom-7b1|auto|185.03|13.35|4.70
11|bigscience/bloom-3b|cuda:0|42.69|55.45|4.95
12|bigscience/bloomz-3b|cuda:0|2.07|1145.38|4.95
13|ehartford/WizardLM-7B-Uncensored|auto|411.75|8.27|59.58
14|LLMs/Vicuna-EvolInstruct-7B|auto|392.41|7.35|32.16

## Language and Text-to-Speech Audio support

* These variables control TTS audio output generator and few other locale-specifics such as initial prompt for chat mode.
* Currently Spanish and English are supported. 
* Straightforward extend to other languages.
Many of these opensource models are trained with several languages, feel free to add yours.

```
LANG = "en" # [en, es]
AUDIO_ENABLED = 0 
```

## Disclaimer

Please don't use for unethical use. Note always licensing terms from the model your are using. Reference and give credits.

## License and copyright 

Please send requests or comments to ecastera@gmail.com

© Enrique Castera Garcia licensed under the [MIT License](LICENSE)








