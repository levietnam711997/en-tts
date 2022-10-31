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
from nltk import sent_tokenize,word_tokenize
import re
nltk.download('punkt')

MAX_LENGTH=20
MAX_WAV_VALUE = 32768.0
device=torch.device('gpu'if torch.cuda.is_available() else 'cpu')
print(device)

#load acoustic model
tacotron_config_file='acoustic/tacotron2_config.json'
tacotron_ckpt_file='acoustic/tacotron2_statedict.pt'
acoustic_config=AttrDict(json.load(open(tacotron_config_file,'r')))
acoustic_config.n_symbols=len(symbols)
acoustic_config.ignore_layers=['embedding.weight']
acoustic_config.text_cleaners=['english_cleaners']
acoustic_model=Tacotron2(acoustic_config)
acoustic_model.eval()
acoustic_model.load_state_dict(torch.load(tacotron_ckpt_file,map_location=device)['state_dict'])

def norm_text(text):
  norm_text=re.sub('\s+',' ',text.strip().lower())
  return norm_text

#load vocoder model
hifigan_config_file="vocoder/config.json"
hifigan_ckpt_file="vocoder/generator_v1"
vocoder_config=AttrDict(json.load(open(hifigan_config_file,'r')))
vocoder_model=Generator(vocoder_config)
vocoder_model.eval()
vocoder_model.load_state_dict(torch.load(hifigan_ckpt_file,map_location=device)['generator'])

def text_to_speech(text):

  sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
  sequence = torch.from_numpy(sequence).to(device).to(torch.long)
  with torch.no_grad():
    print(f'STARTING ENCODE {text}')
    mel_outputs, _, _, _ = acoustic_model.inference(sequence)
    print("STARTING GENERATE SPEAKER")
    y_g_hat=vocoder_model(mel_outputs)
    audio = y_g_hat.squeeze()
    audio = audio * MAX_WAV_VALUE
    audio = audio.cpu().numpy().astype('int16')
    return audio

def speaker(text,output_filename):
  audios=[]
  paragraphs=text.split("\n")
  for paragraph in paragraphs:
    norm_paragraph=norm_text(paragraph)
    sentences=sent_tokenize(norm_paragraph)
    for sentence in sentences:
      sentence_=sentence.strip()
      if sentence_=='':
        continue
      words=word_tokenize(sentence_)
      if len(words)<MAX_LENGTH:
        audio=text_to_speech(sentence_)
        audios.append(audio)
      else:
        for seg in range(0,len(words),MAX_LENGTH):
          audio=text_to_speech(" ".join(words[seg:seg+MAX_LENGTH]))
          audios.append(audio)
  audios=np.concatenate(audios,axis=0)
  write(output_filename, vocoder_config.sampling_rate, audios)

if __name__=='__main__':
  text="""Yesterday I saw a guy spill all his Scrabble letters on the road. I asked him, “What’s the word on the street?”"""
  speaker(text,"output.wav")

