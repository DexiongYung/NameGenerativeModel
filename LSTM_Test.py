import argparse
import json
from collections import OrderedDict
import torch
from Utilities.Convert import *
from Models.Decoder import Decoder

parser = argparse.ArgumentParser()
parser.add_argument('--config', nargs='?', default='Config/first.json')
parser.add_argument('--length', nargs='?', default=5, type=int)
parser.add_argument('--sample', nargs='?', default=25, type=int)
args = parser.parse_args()
def load_json(jsonpath: str) -> dict:
    with open(jsonpath) as jsonfile:
        return json.load(jsonfile, object_pairs_hook=OrderedDict)

def sample(length: list):
    with torch.no_grad():
        max_length = length[0]
        lstm_input = indexTensor([[SOS]], 1, IN_CHARS).to(DEVICE)
        lng_input = lengthTestTensor([length]).to(DEVICE)
        lstm_hidden = lstm.initHidden(1)
        lstm_hidden = (lstm_hidden[0].to(DEVICE), lstm_hidden[1].to(DEVICE))
        name = ''
        char = SOS

        for i in range(max_length):
            lstm_probs, lstm_hidden = lstm(
                lstm_input[0], lng_input, lstm_hidden)

            lstm_probs = lstm_probs.reshape(OUT_COUNT)
            lstm_probs[OUT_CHARS.index(EOS)] = float("-inf")
            lstm_probs[OUT_CHARS.index(PAD)] = float("-inf")
            lstm_probs = lstm_probs.exp()

            sample = int(torch.distributions.categorical.Categorical(
                lstm_probs).sample().item())
            char = OUT_CHARS[sample]
            name += char
            
            lstm_input = indexTensor([[char]], 1, IN_CHARS).to(DEVICE)

        return name
config = load_json(args.config)

SOS = config['SOS']
EOS = config['EOS']
PAD = config['PAD']
IN_CHARS = config['input']
OUT_CHARS = config['output']
IN_COUNT = len(IN_CHARS)
OUT_COUNT = len(OUT_CHARS)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

name = config['session_name']
lstm = Decoder(config['input_sz'], config['hidden_size'], config['output_sz'], padding_idx=config['input'].index(config['PAD']), num_layers=config['num_layers'], embed_size=config['embed_dim'])
lstm.load_state_dict(torch.load(f'Checkpoints/{name}.path.tar', map_location=torch.device('cpu'))['weights'])

for _ in range(args.sample):
    print(sample([args.length]))
