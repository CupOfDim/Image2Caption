from PIL import Image
import numpy as np
import string


def prep_image(data_folder, img, idx, desired_size, var):
    img = img.resize(desired_size, Image.LANCZOS)
    img = np.array(img)
    img = Image.fromarray(img)
    output_path = f"{data_folder}/{var}_img/img_{idx}.jpg"
    img.save(output_path)
    return output_path


def remove_punct(text_orig):
  text = text_orig.translate(string.punctuation)
  return text


def remove_single_char(text_orig):
  text = []
  for word in text_orig.split():
    if len(word) > 1:
      text.append(word)
  return " ".join(text)


def remove_num(text_orig):
  text = []
  for word in text_orig.split():
    if word.isalpha():
      text.append(word)
  return " ".join(text)


def prep_text(text_orig):
  text = text_orig.lower()
  text = remove_punct(text)
  text = remove_single_char(text)
  text = remove_num(text)
  return text


def prep_dataset(data_folder, name_dataset, size, var):
  dataset = {'image_paths': [],
              'captions': []
              }
  for i, d in enumerate(name_dataset):
    image_path = prep_image(data_folder, d['image'], i, size, var)
    c_0 = prep_text(d['caption_0'])
    c_1 = prep_text(d['caption_1'])
    c_2 = prep_text(d['caption_2'])
    c_3 = prep_text(d['caption_3'])
    c_4 = prep_text(d['caption_4'])
    dataset['image_paths'].append(image_path)
    dataset['captions'].append([c_0,c_1,c_2,c_3,c_4])
  return dataset


def add_padding(caption, word_map, max_seq_len):
    cur_seq_len = len(caption)-2
    padd_token = caption + [word_map['<pad>']] * (max_seq_len-cur_seq_len)
    return padd_token


def caption_to_token(dataset, word_map, max_seq_len):
    res = []
    for data in dataset:
        tokens = []
        for caption in data:
            words = caption.split()
            token_caption = [word_map['<start>']] + [word_map[word] for word in words] + [word_map['<end>']]
            token_caption = add_padding(token_caption,word_map, max_seq_len)
            tokens.append(token_caption)
        res.append(tokens)
    return res


def caption_lengths(dataset, word_map):
    res = []
    for data in dataset:
        lens = []
        for caption in data:
            words = caption.split()
            cap_len = len(words)+2
            lens.append(cap_len)
        res.append(lens)
    return res



