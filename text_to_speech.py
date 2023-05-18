# https://github.com/snakers4/silero-models#text-to-speech
import torch
from IPython.display import Audio, display
import numpy as np
from scipy.io.wavfile import write
import os


class TTSModelCard:
    def __init__(self, language, sample_rate, model_id, speaker):
        self.language = language
        self.sample_rate = sample_rate
        self.model_id = model_id
        self.speaker = speaker


tts_models_list = {
    "en_0": TTSModelCard("en", 48000, "v3_en", "en_0"),
    "en_1": TTSModelCard("en", 48000, "v3_en", "en_1"),
    "es_0": TTSModelCard("es", 48000, "v3_es", "es_1"),
    "es_1": TTSModelCard("es", 48000, "v3_es", "es_2")
}


device = torch.device("cpu")


def tts_get_models():
    return list(tts_models_list.keys())


def tts_play(model_id, sample_text):
    modelcard = tts_models_list[model_id]

    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                         model='silero_tts',
                                         language=modelcard.language,
                                         speaker=modelcard.model_id)
    model.to(device)  # gpu or cpu

    print(f"tts_play {model_id} '{sample_text}'")

    audio = model.apply_tts(text=sample_text,
                            speaker=modelcard.speaker,
                            sample_rate=modelcard.sample_rate)

    write('audio.wav', modelcard.sample_rate, np.array(audio))

    #cmd="play audio.wav echo 0.5 0.5 20.0 0.5"
    cmd = "play audio.wav"
    os.system(cmd)


def main():
    sample = "The secret agents who work in big technology companies such as Facebook orGoogle have a diverse range of missions. Some engage in surveillance and cybersecurity, monitoring online activity for foreign governments. Others work on projects related to information warfare, intelligence sharing, and artificial intelligence. Others still may be involved in developing technologies and providing support for operations. Ultimately, their mission is to protect the nation and provide intelligence services to our leaders."
    model = "en_1"

    sample = "Esto es un ejemplo de síntesis de voz en español, ¿le gusta?"
    model = "es_0"

    tts_play(model, sample)


if __name__ == "__main__":
    main()
