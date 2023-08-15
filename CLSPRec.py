import torch
from torch import nn

import settings

device = settings.gpuId if torch.cuda.is_available() else 'cpu'


class CheckInEmbedding(nn.Module):
    def __init__(self, f_embed_size, vocab_size):
        super().__init__()
        self.embed_size = f_embed_size
        poi_num = vocab_size["POI"]
        cat_num = vocab_size["cat"]
        user_num = vocab_size["user"]
        hour_num = vocab_size["hour"]
        day_num = vocab_size["day"]

        self.poi_embed = nn.Embedding(poi_num + 1, self.embed_size, padding_idx=poi_num)
        self.cat_embed = nn.Embedding(cat_num + 1, self.embed_size, padding_idx=cat_num)
        self.user_embed = nn.Embedding(user_num + 1, self.embed_size, padding_idx=user_num)
        self.hour_embed = nn.Embedding(hour_num + 1, self.embed_size, padding_idx=hour_num)
        self.day_embed = nn.Embedding(day_num + 1, self.embed_size, padding_idx=day_num)

    def forward(self, x):
        poi_emb = self.poi_embed(x[0])
        cat_emb = self.cat_embed(x[1])
        user_emb = self.user_embed(x[2])
        hour_emb = self.hour_embed(x[3])
        day_emb = self.day_embed(x[4])

        return torch.cat((poi_emb, cat_emb, user_emb, hour_emb, day_emb), 1)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = self.embed_size // self.heads

        assert (
                self.head_dim * self.heads == self.embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.keys = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.queries = nn.Linear(self.embed_size, self.embed_size, bias=False)
        self.fc_out = nn.Linear(self.heads * self.head_dim, self.embed_size)

    def forward(self, values, keys, query):
        value_len, key_len, query_len = values.shape[0], keys.shape[0], query.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        values = values.reshape(value_len, self.heads, self.head_dim)
        keys = keys.reshape(key_len, self.heads, self.head_dim)
        queries = queries.reshape(query_len, self.heads, self.head_dim)

        energy = torch.einsum("qhd,khd->hqk", [queries, keys])

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=2)

        out = torch.einsum("hql,lhd->qhd", [attention, values]).reshape(
            query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)

        return out


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.embed_size = embed_size
        self.attention = SelfAttention(self.embed_size, heads)
        self.norm1 = nn.LayerNorm(self.embed_size)
        self.norm2 = nn.LayerNorm(self.embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embed_size, forward_expansion * self.embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * self.embed_size, self.embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        attention = self.attention(value, key, query)  # [len * embed_size]

        # Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embedding_layer,
            embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout,
    ):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = embedding_layer
        self.add_module('embedding', self.embedding_layer)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    embed_size,
                    num_heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, feature_seq):
        embedding = self.embedding_layer(feature_seq)  # [len, embedding]
        out = self.dropout(embedding)

        # In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case
        for layer in self.layers:
            out = layer(out, out, out)

        return out


# Attention for query and key with different dimension
class Attention(nn.Module):
    def __init__(
            self,
            qdim,
            kdim,
    ):
        super().__init__()

        # Resize q's dimension to k
        self.expansion = nn.Linear(qdim, kdim)

    def forward(self, query, key, value):
        q = self.expansion(query)  # [embed_size]
        temp = torch.inner(q, key)
        weight = torch.softmax(temp, dim=0)  # [len, 1]
        weight = torch.unsqueeze(weight, 1)
        temp2 = torch.mul(value, weight)
        out = torch.sum(temp2, 0)  # sum([len, embed_size] * [len, 1])  -> [embed_size]

        return out


class CLSPRec(nn.Module):
    def __init__(
            self,
            vocab_size,
            f_embed_size=60,
            num_encoder_layers=1,
            num_lstm_layers=1,
            num_heads=1,
            forward_expansion=2,
            dropout_p=0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.total_embed_size = f_embed_size * 5

        # Layers
        self.embedding = CheckInEmbedding(
            f_embed_size,
            vocab_size
        )
        self.encoder = TransformerEncoder(
            self.embedding,
            self.total_embed_size,
            num_encoder_layers,
            num_heads,
            forward_expansion,
            dropout_p,
        )
        self.lstm = nn.LSTM(
            input_size=self.total_embed_size,
            hidden_size=self.total_embed_size,
            num_layers=num_lstm_layers,
            dropout=0
        )
        self.final_attention = Attention(
            qdim=f_embed_size,
            kdim=self.total_embed_size
        )
        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, self.total_embed_size * forward_expansion),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(self.total_embed_size * forward_expansion, vocab_size["POI"]))

        self.loss_func = nn.CrossEntropyLoss()

        self.tryone_line2 = nn.Linear(self.total_embed_size, f_embed_size)
        self.enhance_val = nn.Parameter(torch.tensor(0.5))

    def feature_mask(self, sequences, mask_prop):
        masked_sequences = []
        for seq in sequences:  # each long term sequences
            feature_seq, day_nums = seq[0], seq[1]
            seq_len = len(feature_seq[0])
            mask_count = torch.ceil(mask_prop * torch.tensor(seq_len)).int()
            masked_index = torch.randperm(seq_len - 1) + torch.tensor(1)
            masked_index = masked_index[:mask_count]  # randomly generate mask index

            feature_seq[0, masked_index] = self.vocab_size["POI"]  # mask POI
            feature_seq[1, masked_index] = self.vocab_size["cat"]  # mask cat
            feature_seq[3, masked_index] = self.vocab_size["hour"]  # mask hour
            feature_seq[4, masked_index] = self.vocab_size["day"]  # mask day

            masked_sequences.append((feature_seq, day_nums))
        return masked_sequences

    def ssl(self, embedding_1, embedding_2, neg_embedding):
        def score(x1, x2):
            return torch.mean(torch.mul(x1, x2))

        def single_infoNCE_loss_simple(embedding1, embedding2, neg_embedding):
            pos = score(embedding1, embedding2)
            neg1 = score(embedding1, neg_embedding)
            neg2 = score(embedding2, neg_embedding)
            neg = (neg1 + neg2) / 2
            one = torch.cuda.FloatTensor([1], device=device)
            con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos)) - torch.log(1e-8 + (one - torch.sigmoid(neg))))
            return con_loss

        ssl_loss = single_infoNCE_loss_simple(embedding_1, embedding_2, neg_embedding)
        return ssl_loss

    def forward(self, sample, neg_sample_list):
        # Process input sample
        long_term_sequences = sample[:-1]
        short_term_sequence = sample[-1]
        short_term_features = short_term_sequence[0][:, :- 1]
        target = short_term_sequence[0][0, -1]
        user_id = short_term_sequence[0][2, 0]

        # Random mask long-term sequences
        long_term_sequences = self.feature_mask(long_term_sequences, settings.mask_prop)

        # Long-term
        long_term_out = []
        for seq in long_term_sequences:
            output = self.encoder(feature_seq=seq[0])
            long_term_out.append(output)
        long_term_catted = torch.cat(long_term_out, dim=0)

        # Short-term
        short_term_state = self.encoder(feature_seq=short_term_features)

        # User enhancement
        user_embed = self.embedding.user_embed(user_id)
        embedding = torch.unsqueeze(self.embedding(short_term_features), 0)
        output, _ = self.lstm(embedding)
        short_term_enhance = torch.squeeze(output)
        user_embed = self.enhance_val * user_embed + (1 - self.enhance_val) * self.tryone_line2(
            torch.mean(short_term_enhance, dim=0))

        # SSL
        neg_short_term_states = []
        for neg_day_sample in neg_sample_list:
            neg_trajectory_features = neg_day_sample[0]
            neg_short_term_state = self.encoder(feature_seq=neg_trajectory_features)
            neg_short_term_state = torch.mean(neg_short_term_state, dim=0)
            neg_short_term_states.append(neg_short_term_state)

        short_embed_mean = torch.mean(short_term_state, dim=0)
        long_embed_mean = torch.mean(long_term_catted, dim=0)
        neg_embed_mean = torch.mean(torch.stack(neg_short_term_states), dim=0)
        ssl_loss = self.ssl(short_embed_mean, long_embed_mean, neg_embed_mean)

        # Final predict
        h_all = torch.cat((short_term_state, long_term_catted))
        final_att = self.final_attention(user_embed, h_all, h_all)
        output = self.out_linear(final_att)

        label = torch.unsqueeze(target, 0)
        pred = torch.unsqueeze(output, 0)

        pred_loss = self.loss_func(pred, label)
        loss = pred_loss + ssl_loss * settings.neg_weight
        return loss, output

    def predict(self, sample, neg_sample_list):
        _, pred_raw = self.forward(sample, neg_sample_list)
        ranking = torch.sort(pred_raw, descending=True)[1]
        target = sample[-1][0][0, -1]

        return ranking, target
