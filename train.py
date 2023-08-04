from mmodels import Encoder, Decoder
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import corpus_bleu
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from tqdm.notebook import tqdm
from torch import nn
from utils import *
import time
import pickle

data_folder = ''
with open(f'{data_folder}/word_map.json', 'r') as file:
    word_map = json.load(file)
train_folder = f'{data_folder}/train_data.json'
test_folder = f'{data_folder}/test_data.json'
val_folder = f'{data_folder}/validation_data.json'

emb_dim = 512
attention_dim = 512
decoder_dim = 512
drop_out = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cudnn.benchmark = True
vocab_size = len(word_map)

batch_size = 32
start = 0
epoches = 150
not_improve_epoches = 0
encoder_lr = 1e-4
decoder_lr = 1e-3
grad_clip = 1.
alpha_c = 1.
best_bleu4 = 0.
encoder_fine_tune = False
checkpoint = None
print_info = 100

if checkpoint is None:
    decoder = Decoder(decoder_dim, attention_dim, emb_dim, vocab_size)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=decoder_lr)

    encoder = Encoder()
    encoder.fine_tune(encoder_fine_tune)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=encoder_lr) if encoder_fine_tune else None

else:
    checkpoint = torch.load(checkpoint)
    start = checkpoint['epoch'] + 1
    not_improve_epoches = checkpoint['not_improve_epoches']
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder_optimizer = checkpoint['encoder_optimizer']
    best_bleu4 = checkpoint['bleu4']

    if encoder_fine_tune and encoder_optimizer is None:
        encoder.fine_tune(encoder_fine_tune)
        encoder_optimizer = optim.Adam(params=[p for p in encoder.parameters() if p.requires_grad], lr=encoder_lr)

decoder = decoder.to(device)
encoder = encoder.to(device)

criterion = nn.CrossEntropyLoss().to(device)

normalize = transforms.Normalize(mean=[0.480, 0.450, 0.400], std=[0.230, 0.225, 0.225])
train_loader = torch.utils.data.DataLoader(CaptionDatasets(train_folder, 'TRAIN', transform=transforms.Compose([normalize])), batch_size=batch_size, shuffle = True, drop_last=True)
val_loader = torch.utils.data.DataLoader(CaptionDatasets(val_folder, 'VAL', transform=transforms.Compose([normalize])), batch_size=batch_size, shuffle = True, drop_last=True)

with open(f'{data_folder}/train_loader.pkl', 'wb') as file:
    pickle.dump(train_loader, file)
with open(f'{data_folder}/val_loader.pkl', 'wb') as file:
    pickle.dump(val_loader, file)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    encoder.train()
    decoder.train()

    losses = AverageMeter()

    start = time.time()

    for i, (images, caps, caplens) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        enc_out = encoder(images)
        scores, alphas, captions, decode_lens, sort_ind = decoder(enc_out, caps, caplens)

        targets = torch.stack([cap[1:] for cap in captions])
        packed_scores = pack_padded_sequence(scores, decode_lens, batch_first=True)
        scores, _ = pad_packed_sequence(packed_scores, batch_first=True)
        packed_targets = pack_padded_sequence(targets, decode_lens, batch_first=True)
        targets, _ = pad_packed_sequence(packed_targets, batch_first=True)
        loss = criterion(scores.float().permute(0, 2, 1), targets)
        loss += alpha_c * ((1 - alphas.sum(dim=1))**2).mean()

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        losses.update(loss.item(), sum(decode_lens))

        if i % 100 == 0:
            print(f'Epoch {epoch} : {i}/{len(train_loader)}\n Loss: {losses.val} avg loss: {losses.avg}')
    finish = time.time() - start
    print(f'Time: {finish:.2f} sec')


def valid(val_loader, encoder, decoder, criterion, epoch, word_map):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    losses = AverageMeter()

    start = time.time()

    refs = list()
    hypoths = list()

    with torch.no_grad():
        for i, (images, caps, caplens, all_caps) in tqdm(enumerate(val_loader)):
            images = images.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            all_caps = all_caps.to(device)

            if encoder is not None:
                images = encoder(images)
            scores, alphas, captions, decode_lens, sort_ind = decoder(images, caps, caplens)

            targets = torch.stack([cap[1:] for cap in captions])

            scores_copy = scores.clone()
            packed_scores = pack_padded_sequence(scores, decode_lens, batch_first=True)
            scores, _ = pad_packed_sequence(packed_scores, batch_first=True)
            packed_targets = pack_padded_sequence(targets, decode_lens, batch_first=True)
            targets, _ = pad_packed_sequence(packed_targets, batch_first=True)

            loss = criterion(scores.float().permute(0, 2, 1), targets)
            loss += alpha_c * ((1 - alphas.sum(dim=1))**2).mean()

            losses.update(loss, sum(decode_lens))

            if i % print_info == 100:
                print(f'Epoch {epoch}\n Val loss: {losses.val} avg val loss: {losses.avg}')

            all_caps = all_caps[sort_ind]
            for j in range(all_caps.shape[0]):
                img_caps = all_caps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                refs.append(img_captions)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for i, p in enumerate(preds):
                temp_preds.append(p[:decode_lens[i]])
            preds = temp_preds
            hypoths.extend(preds)

            assert len(refs) == len(hypoths)
        finish = time.time() - start
        print(f'Time: {finish:.2f} sec')
        bleu4 = corpus_bleu(refs,hypoths)

        print(f'Loss: {losses.val:.3f} Bleu: {bleu4:.3f}')
    return bleu4


for epoch in tqdm(range(start, epoches)):
    if not_improve_epoches == 20:
        break
    if not_improve_epoches > 0 and not_improve_epoches%8 == 0:
        adjust_learning_rate(decoder_optimizer, 0.8)
        if encoder_fine_tune:
            adjust_learning_rate(encoder_optimizer, 0.8)

    train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch)
    rec_bleu4 = valid(val_loader, encoder, decoder, criterion, epoch, word_map)

    is_best = rec_bleu4 > best_bleu4
    best_bleu4 = max(best_bleu4, rec_bleu4)
    if not is_best:
        not_improve_epoches += 1
    else:
        not_improve_epoches = 0
    save_checkpoint(data_folder, 'flickr8', epoch, not_improve_epoches, encoder, decoder, encoder_optimizer, decoder_optimizer, rec_bleu4, is_best)