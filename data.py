import json
from datasets import load_dataset
from data_prepare_functions import prep_dataset, prep_text, caption_to_token, caption_lengths

dataset = load_dataset("jxie/flickr8k")
data_folder = ''  #here data file path

size = (256, 256)
prep_train_dataset = prep_dataset(data_folder, dataset['train'], size, 'train')
prep_test_dataset = prep_dataset(data_folder, dataset['test'], size, 'test')
prep_val_dataset = prep_dataset(data_folder, dataset['validation'], size, 'validation')

vocab = []
seq = []
for v in ['train', 'test', 'validation']:
    for d in dataset[v]:
        for i in range(5):
            caption = prep_text(d[f'caption_{i}'])
            vocab.extend(caption.split())
            seq.append(caption.split())

vocab_size = len(set(vocab))
vocab = set(vocab)
max_seq_len = len(max(seq, key=len))+1

word_map = {k: v+1 for v, k in enumerate(vocab)}
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0

with open(f'{data_folder}/word_map.json', 'w') as file:
    json.dump(word_map, file)

prep_train_dataset['tokens'] = caption_to_token(prep_train_dataset['captions'],word_map, max_seq_len)
prep_train_dataset['lens'] = caption_lengths(prep_train_dataset['captions'], word_map)

prep_test_dataset['tokens'] = caption_to_token(prep_test_dataset['captions'],word_map, max_seq_len)
prep_test_dataset['lens'] = caption_lengths(prep_test_dataset['captions'], word_map)

prep_val_dataset['tokens'] = caption_to_token(prep_val_dataset['captions'],word_map, max_seq_len)
prep_val_dataset['lens'] = caption_lengths(prep_val_dataset['captions'],word_map)

for var in ['train', 'test', 'validation']:
    with open(f'{data_folder}/{var}_data.json', 'w') as file:
        if var == 'train':
            json.dump(prep_train_dataset, file, indent=4)
        elif var == 'test':
            json.dump(prep_test_dataset, file, indent=4)
        else:
            json.dump(prep_val_dataset, file, indent=4)