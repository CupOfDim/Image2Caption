import torch
import torchvision
from torch import nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, img_size=14):
        super(Encoder, self).__init__()
        resnet = torchvision.models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.ad_pool = nn.AdaptiveAvgPool2d((img_size, img_size))

        self.fine_tune()

    def fine_tune(self, fine_tune=True):
        for p in self.resnet.parameters():
            p.requires_grad = False
        for c in self.resnet.children():
            for p in c.parameters():
                p.requires_grad = fine_tune

    def forward(self, images):
        out = self.resnet(images)
        out = self.ad_pool(out)
        out = out.permute(0, 2, 3, 1)
        return out


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encode_attn = nn.Linear(encoder_dim, attention_dim)
        self.decode_attn = nn.Linear(decoder_dim, attention_dim)
        self.f_attn = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encode_out, decode_hidden):
        attn1 = self.encode_attn(encode_out)
        attn2 = self.decode_attn(decode_hidden)
        f_attn = self.f_attn(self.relu(attn1 + attn2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(f_attn)
        attention_weights = (encode_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weights, alpha


class Decoder(nn.Module):
    def __init__(self, decode_dim, attention_dim, embedding_dim, vocab_size, encode_dim=2048, dropout=0.5):
        super(Decoder, self).__init__()
        self.encode_dim = encode_dim
        self.decode_dim = decode_dim
        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encode_dim, decode_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decoder = nn.LSTMCell(embedding_dim + encode_dim, decode_dim, bias=True)

        self.h_d = nn.Linear(encode_dim, decode_dim)
        self.c_d = nn.Linear(encode_dim, decode_dim)
        self.beta = nn.Linear(decode_dim, encode_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decode_dim, vocab_size)
        self.init_weight()

    def init_weight(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)

    def load_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encode_out):
        mean_encode_out = encode_out.mean(dim=1)#(64,64)
        h = self.h_d(mean_encode_out)
        c = self.c_d(mean_encode_out)
        return h, c

    def forward(self, encode_out, encode_captions, caption_length):
        batch_size = encode_out.size(0)#64
        encode_dim = encode_out.size(-1)#64
        vocab_size = self.vocab_size#8253

        encode_out = encode_out.view(batch_size, -1, encode_dim)#(64, 196, 64)
        num_pixels = encode_out.size(1)

        caption_length, sort_ind = caption_length.squeeze(1).sort(dim=0, descending=True)

        encode_captions = encode_captions[sort_ind]
        encode_out = encode_out[sort_ind]

        decode_lens = (caption_length-1).tolist()

        embeddings = self.embedding(encode_captions)

        h, c = self.init_hidden_state(encode_out)

        predictions = torch.zeros(batch_size, max(decode_lens), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lens), num_pixels).to(device)

        for t in range(max(decode_lens)):
            batch_size_t = sum([l > t for l in decode_lens])
            attention_weights, alpha = self.attention(encode_out[:batch_size_t], h[:batch_size_t])  #attn:(batch, encode_dim), alpha:(batch, num_pixels)
            gate = self.sigmoid(self.beta(h[:batch_size_t]))
            attention_weights = gate * attention_weights

            h, c = self.decoder(torch.cat([embeddings[:batch_size_t, t, :], attention_weights], dim=1),
                                       (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        return predictions, alphas, encode_captions, decode_lens, sort_ind

