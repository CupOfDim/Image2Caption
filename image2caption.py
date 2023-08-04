import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_folder = ''


def caption_beam_search(encoder, decoder, img_path, word_map, beam_size=3):
    k = beam_size
    vocab = word_map

    img = Image.open(img_path)
    img = img.resize((256, 256), Image.LANCZOS)
    img = np.array(img)
    img = img.transpose(2, 0, 1)
    img = img / 255
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.480, 0.450, 0.400], std=[0.230, 0.225, 0.225])
    transform = transforms.Compose([normalize])

    image = transform(img)

    # image = image.unsqueeze(0)
    encoder_out = encoder(image)
    image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(-1)

    encoder_out = encoder_out.view(1, -1, encoder_dim)

    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)

    seqs = k_prev_words

    top_k_scores = torch.zeros(k, 1).to(device)
    seqs_alpha = torch.ones(k, 1, image_size, image_size).to(device)

    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    while True:
        embeddings = decoder.embedding(k_prev_words)
        aw, alpha = decoder.attention(encoder_out, h)
        alpha = alpha.view(-1, image_size, image_size)
        gate = decoder.sigmoid(decoder.beta(h))
        aw = gate * aw
        print(h.size(), c.size(), aw.size(), embeddings.size())
        h, c = decoder.decoder(torch.cat([embeddings, aw], dim=1), (h,c))

        scores = decoder.fc(h)
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)

        prev_word_inds = top_k_words / vocab
        next_word_inds = top_k_words % vocab

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != 102]
        complete_inds = list(set(range(len(next_word_inds)))) - set(incomplete_inds)

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    return seq, alphas


def visual_attn(image_path, seq, ind_word_map, alphas, smooth=True):
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [ind_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    checkpoint = torch.load(f'{data_folder}/BEST_checkpoint_flickr8.pth.tar')
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    # Load word map (word2ix)
    with open(f'{data_folder}/word_map.json', 'r') as file:
        word_map = json.load(file)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word
    img_path = f'{data_folder}/test_img/img_0.jpg'
    # Encode, decode with attention and beam search
    seq, alphas = caption_beam_search(encoder, decoder, img_path, word_map, 3)
    alphas = torch.FloatTensor(alphas)

    # Visualize caption and attention of best sequence
    visual_attn(img_path, seq, alphas, rev_word_map)