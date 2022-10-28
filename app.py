import torch
from acoustic_model import Tacotron2
import numpy as np
from utils import text_to_sequence
from utils import init_weights
from vocoder import Generator
import json
from env import AttrDict
from scipy.io.wavfile import write
from symbols import symbols
import nltk
from nltk import sent_tokenize

nltk.download('punkt')

MAX_WAV_VALUE = 32768.0
device=torch.device('gpu'if torch.cuda.is_available() else 'cpu')

#load acoustic model
tacotron_config_file='tacotron2_config.json'
tacotron_ckpt_file='tacotron2_statedict.pt'
acoustic_config=AttrDict(json.load(open(tacotron_config_file,'r')))
acoustic_config.n_symbols=len(symbols)
acoustic_config.ignore_layers=['embedding.weight']
acoustic_config.text_cleaners=['english_cleaners']
acoustic_model=Tacotron2(acoustic_config)
acoustic_model.eval()
acoustic_model.load_state_dict(torch.load(tacotron_ckpt_file,map_location=device)['state_dict'])



#load vocoder model
hifigan_config_file="LJ_FT_T2_V1/config.json"
hifigan_ckpt_file="LJ_FT_T2_V1/generator_v1"
vocoder_config=AttrDict(json.load(open(hifigan_config_file,'r')))
vocoder_model=Generator(vocoder_config)
vocoder_model.eval()
vocoder_model.load_state_dict(torch.load(hifigan_ckpt_file,map_location=device)['generator'])

def speaker(text):

  sequence = np.array(text_to_sequence(text.strip().lower(), ['english_cleaners']))[None, :]
  sequence = torch.from_numpy(sequence).to(device).to(torch.long)
  with torch.no_grad():
    mel_outputs, _, _, _ = acoustic_model.inference(sequence)
    y_g_hat=vocoder_model(mel_outputs)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio

if __name__=='__main__':
  text="""Yesterday I saw a guy spill all his Scrabble letters on the road. I asked him, “What’s the word on the street?”"""
  sentences=sent_tokenize(text)
  audios=[]
  for sentence in sentences:
    audio=speaker(sentence)
    audios.append(audio)
  audios=np.concatenate(audios,axis=0)
  write("output.wav", vocoder_config.sampling_rate, audios)


