from abc import ABC
from typing import Optional
from torch import nn
from allennlp.modules.conditional_random_field import ConditionalRandomField
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput
from transformers.modeling_utils import ModuleUtilsMixin, GenerationMixin


def get_angles(pos, i, H):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(H))
    value = pos * angle_rates

    return value


def get_sinusoidal_score(max_sentence_length, H):
    angle_rates = get_angles(np.arange(max_sentence_length)[:, np.newaxis], np.arange(H)[np.newaxis, :], H)

    sines = np.sin(angle_rates[:, 0::2])
    cosines = np.cos(angle_rates[:, 1::2])

    weight = np.concatenate([sines, cosines], axis=-1)
    weight = np.cast[np.float32](weight)

    return weight


def plot_position_embedding(position_embedding):
    plt.pcolormesh(position_embedding, cmap="RdBu")
    plt.xlabel("Depth")
    plt.xlim((0, position_embedding.shape[1]))
    plt.ylabel("Position")
    plt.ylim((position_embedding.shape[0], 0))
    plt.colorbar()
    plt.show()


def get_padding_mask(x):
    mask = 1 - np.cast[np.int32](np.equal(x, 0))
    mask = mask[:, np.newaxis, np.newaxis, :]
    mask = torch.tensor(mask, dtype=torch.int32)

    return mask


def get_look_ahead_mask(max_sentence_length):
    mask = np.tril(np.ones((max_sentence_length, max_sentence_length)))
    mask = np.cast[np.int32](mask)
    mask = torch.tensor(mask, dtype=torch.int32)

    return mask


class SinusoidalPositionEmbedding(nn.Embedding, ModuleUtilsMixin):
    def __init__(self, max_sentence_length, embedding_dim, trainable=False, **kwargs):
        _weight = torch.tensor(get_sinusoidal_score(max_sentence_length, embedding_dim), dtype=torch.float32)
        super(SinusoidalPositionEmbedding, self).__init__(
            num_embeddings=max_sentence_length,
            embedding_dim=embedding_dim,
            _weight=_weight,
            **kwargs)
        self.weight.requires_grad = trainable


class TrainablePositionEmbedding(nn.Embedding, ModuleUtilsMixin):
    def __init__(self, max_sentence_length, embedding_dim, **kwargs):
        super(TrainablePositionEmbedding, self).__init__(max_sentence_length, embedding_dim, **kwargs)


class PositionAwareEmbedding(nn.Module, ModuleUtilsMixin):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 max_sentence_length,
                 dropout_rate,
                 position_embedding_trainable,
                 position_embedding_layer):
        super(PositionAwareEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sentence_length = max_sentence_length
        self.position_embedding_trainable = position_embedding_trainable

        self.token_embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = position_embedding_layer(
            max_sentence_length,
            embedding_dim,
            position_embedding_trainable)
        self.segment_embedding = torch.nn.Embedding(2, embedding_dim)

        self.layer_norm = torch.nn.LayerNorm(embedding_dim, eps=1e-12)
        self.dropout_layer = torch.nn.Dropout(dropout_rate)

    def forward(self, x, x_type_ids=None, x_position_ids=None):
        batch_size = x.shape[0]
        input_seq_len = x.shape[1]
        assert input_seq_len <= self.max_sentence_length

        tok_emb = self.token_embedding(x)
        tok_emb *= torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32))

        if x_position_ids is None:
            x_position_ids = torch.tensor([np.arange(input_seq_len) for _ in range(batch_size)])
        x_position_ids = x_position_ids.to(x.device)
        pos_emb = self.position_embedding(x_position_ids)

        if x_type_ids is None:
            x_type_ids = torch.zeros((batch_size, input_seq_len), dtype=torch.long)
        x_type_ids = x_type_ids.to(x.device)
        seg_emb = self.segment_embedding(x_type_ids)

        emb = tok_emb + pos_emb + seg_emb
        hidden = self.layer_norm(emb)
        output = self.dropout_layer(hidden)

        return output


class ScaledDotProductAttention(nn.Module, ModuleUtilsMixin):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, q, k, v, mask):
        dim_q = q.shape[-1]
        logit = torch.matmul(q, torch.transpose(k, -2, -1)) / torch.sqrt(torch.tensor(dim_q, dtype=torch.float32))

        if mask is not None:
            logit = logit.masked_fill(mask == 0, 1e-9)

        attention = torch.softmax(logit, -1)
        output = torch.matmul(attention, v)

        return output, attention


class MultiHeadAttention(nn.Module, ModuleUtilsMixin):
    def __init__(self, H, A, dropout_rate, attention_head, output_attentions=False):
        super(MultiHeadAttention, self).__init__()
        assert H % A == 0
        self.H = H
        self.A = A
        self.depth = H // A
        self.attention_head = attention_head()
        self.output_attentions = output_attentions

        self.W_q = torch.nn.Linear(self.H, self.H)
        self.W_k = torch.nn.Linear(self.H, self.H)
        self.W_v = torch.nn.Linear(self.H, self.H)
        self.dense_layer = torch.nn.Linear(self.H, self.H)

        self.dropout_layer = torch.nn.Dropout(dropout_rate)

    def split_heads(self, x):
        batch_size = x.shape[0]
        x = torch.reshape(x, (batch_size, -1, self.A, self.depth))
        x = torch.transpose(x, 1, 2)

        return x

    def forward(self, q, k, v, mask):
        batch_size = q.shape[0]

        q = self.W_q(q)
        q = self.split_heads(q)

        k = self.W_k(k)
        k = self.split_heads(k)

        v = self.W_v(v)
        v = self.split_heads(v)

        split_head_attention_score, split_head_attention_weight = self.attention_head(q, k, v, mask)
        split_head_attention_score = torch.transpose(split_head_attention_score, 1, 2)
        concat_head_attention_score = torch.reshape(split_head_attention_score, (batch_size, -1, self.H))
        concat_head_attention_score = self.dropout_layer(concat_head_attention_score)
        output = self.dense_layer(concat_head_attention_score)

        output = (output,)
        if self.output_attentions:
            output = output + (split_head_attention_weight,)

        return output


class FeedForwardLayer(nn.Module, ModuleUtilsMixin):
    def __init__(self, H, feed_forward_dim, activation_layer):
        super(FeedForwardLayer, self).__init__()
        self.activation_layer = activation_layer()

        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(H, feed_forward_dim),
             self.activation_layer,
             torch.nn.Linear(feed_forward_dim, H)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class TransformerEncoderLayer(nn.Module, ModuleUtilsMixin):
    def __init__(self, H, A,
                 feed_forward_dim,
                 dropout_rate,
                 attention_head,
                 multi_head_attention,
                 activation_layer,
                 feed_forward_layer,
                 output_attentions=False):
        super(TransformerEncoderLayer, self).__init__()
        self.H = H
        self.A = A
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate

        self.multi_head_attention = multi_head_attention(H, A, dropout_rate, attention_head, output_attentions)
        self.feed_forward_layer = feed_forward_layer(H, feed_forward_dim, activation_layer)

        self.layer_norm_1 = torch.nn.LayerNorm(H, eps=1e-6)
        self.layer_norm_2 = torch.nn.LayerNorm(H, eps=1e-6)

        self.dropout_layer_1 = torch.nn.Dropout(dropout_rate)
        self.dropout_layer_2 = torch.nn.Dropout(dropout_rate)

    def forward(self, x, padding_mask=None):
        attention = self.multi_head_attention(x, x, x, padding_mask)
        attended = attention[0]

        hidden = self.layer_norm_1(x + attended)
        middle = self.dropout_layer_1(hidden)

        hidden = self.feed_forward_layer(middle)
        hidden = self.layer_norm_2(middle + hidden)
        output = self.dropout_layer_2(hidden)

        output = (output,) + attention[1:]

        return output


class TransformerDecoderLayer(nn.Module, ModuleUtilsMixin, GenerationMixin, ABC):
    def __init__(self, H, A,
                 feed_forward_dim,
                 dropout_rate,
                 attention_head,
                 multi_head_attention,
                 activation_layer,
                 feed_forward_layer,
                 output_attentions=False):
        super(TransformerDecoderLayer, self).__init__()
        self.H = H
        self.A = A
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate

        self.masked_attention = multi_head_attention(H, A, dropout_rate, attention_head, output_attentions)
        self.encoder_decoder_attention = multi_head_attention(H, A, dropout_rate, attention_head)
        self.feed_forward_layer = feed_forward_layer(H, feed_forward_dim, activation_layer)

        self.layer_norm_1 = torch.nn.LayerNorm(H, eps=1e-6)
        self.layer_norm_2 = torch.nn.LayerNorm(H, eps=1e-6)
        self.layer_norm_3 = torch.nn.LayerNorm(H, eps=1e-6)

        self.dropout_layer_1 = torch.nn.Dropout(dropout_rate)
        self.dropout_layer_2 = torch.nn.Dropout(dropout_rate)
        self.dropout_layer_3 = torch.nn.Dropout(dropout_rate)

    def forward(self, x, encoder_output, look_ahead_mask, padding_mask):
        masked_attention = self.masked_attention(x, x, x, look_ahead_mask)
        masked_attended = masked_attention[0]
        hidden = self.layer_norm_1(x + masked_attended)
        middle = self.dropout_layer_1(hidden)

        encoder_decoder_attention = self.encoder_decoder_attention(middle, encoder_output, encoder_output, padding_mask)
        encoder_decoder_attended = encoder_decoder_attention[0]
        hidden = self.layer_norm_2(middle + encoder_decoder_attended)
        middle = self.dropout_layer_2(hidden)

        hidden = self.feed_forward_layer(middle)
        hidden = self.layer_norm_3(middle + hidden)
        output = self.dropout_layer_3(hidden)

        output = (output, ) + masked_attention[1:] + encoder_decoder_attention[1:]

        return output


class TransformerEncoder(nn.Module, ModuleUtilsMixin):
    def __init__(self,
                 L: int = 3,
                 H: int = 768,
                 A: int = 12,
                 vocab_size: int = 30000,
                 max_sentence_length: int = 512,
                 feed_forward_dim: int = 3072,
                 dropout_rate: float = 0.1,
                 position_embedding_trainable: bool = False,
                 position_embedding_layer=SinusoidalPositionEmbedding,
                 embedding_layer=PositionAwareEmbedding,
                 attention_head=ScaledDotProductAttention,
                 multi_head_attention=MultiHeadAttention,
                 activation_layer=torch.nn.ReLU,
                 feed_forward_layer=FeedForwardLayer,
                 encoder_layer=TransformerEncoderLayer,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False):
        super(TransformerEncoder, self).__init__()
        self.L = L
        self.H = H
        self.A = A
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        self.embedding_layer = embedding_layer(
            vocab_size,
            H,
            max_sentence_length,
            dropout_rate,
            position_embedding_trainable,
            position_embedding_layer
        )

        self.encoder_layers = torch.nn.ModuleList([
            encoder_layer(H, A,
                          feed_forward_dim,
                          dropout_rate,
                          attention_head,
                          multi_head_attention,
                          activation_layer,
                          feed_forward_layer,
                          output_attentions)
            for _ in range(L)
        ])

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None):
        hidden = self.embedding_layer(input_ids, token_type_ids, position_ids)
        hidden_states = (hidden,)

        attentions = ()
        for i in range(len(self.encoder_layers)):
            layer = self.encoder_layers[i]
            encoded = layer(hidden, attention_mask)
            hidden = encoded[0]

            hidden_states += (hidden,)
            attentions += encoded[1:]

        if not self.output_attentions and not self.output_hidden_states:
            return hidden

        output = (hidden,)

        if self.output_hidden_states:
            output += (hidden_states,)

        if self.output_attentions:
            output += (attentions,)

        return output

    def set_output_attentions(self, boolean: bool):
        self.output_attentions = boolean

        for layer in self.encoder_layers:
            layer.multi_head_attention.output_attentions = boolean

    def set_output_hidden_states(self, boolean: bool):
        self.output_hidden_states = boolean


class TransformerDecoder(nn.Module, ModuleUtilsMixin, GenerationMixin, ABC):
    def __init__(self,
                 L: int = 3,
                 H: int = 768,
                 A: int = 12,
                 vocab_size: int = 30000,
                 max_sentence_length: int = 512,
                 feed_forward_dim: int = 3072,
                 dropout_rate: float = 0.1,
                 embedding_binding: bool = True,
                 encoder_embedding_layer: Optional[nn.Module] = None,
                 position_embedding_trainable: bool = False,
                 position_embedding_layer=SinusoidalPositionEmbedding,
                 embedding_layer=PositionAwareEmbedding,
                 attention_head=ScaledDotProductAttention,
                 multi_head_attention=MultiHeadAttention,
                 activation_layer=torch.nn.ReLU,
                 feed_forward_layer=FeedForwardLayer,
                 decoder_layer=TransformerDecoderLayer,
                 output_attentions=False,
                 output_hidden_states=False):
        super(TransformerDecoder, self).__init__()
        self.L = L
        self.H = H
        self.A = A
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate = dropout_rate
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        if embedding_binding and encoder_embedding_layer is not None:
            self.embedding_layer = encoder_embedding_layer
        else:
            self.embedding_layer = embedding_layer(
                vocab_size,
                H,
                max_sentence_length,
                dropout_rate,
                position_embedding_trainable,
                position_embedding_layer
            )

        self.decoder_layers = torch.nn.ModuleList([
            decoder_layer(H, A,
                          feed_forward_dim,
                          dropout_rate,
                          attention_head,
                          multi_head_attention,
                          activation_layer,
                          feed_forward_layer,
                          output_attentions)
            for _ in range(L)
        ])

        self.logit_layer = nn.Linear(H, vocab_size)

    def forward(self,
                input_ids: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None):
        hidden = self.embedding_layer(input_ids, position_ids, token_type_ids)
        hidden_states = (hidden,)

        masked_attentions = ()
        encoder_decoder_attentions = ()
        for i in range(len(self.decoder_layers)):
            layer = self.decoder_layers[i]
            encoded = layer(hidden, encoder_hidden_states, attention_mask, encoder_attention_mask)
            hidden = encoded[0]
            hidden_states += (hidden,)

            if len(encoded) > 1:
                masked_attentions += (encoded[1],)
                encoder_decoder_attentions += (encoded[2],)

        logit = self.logit_layer(hidden)

        if not self.output_attentions and not self.output_attentions:
            return logit

        output = (logit,)

        if self.output_hidden_states:
            output += (hidden_states,)

        if self.output_attentions:
            output += (masked_attentions,) + (encoder_decoder_attentions,)

        return output

    def set_output_attentions(self, boolean: bool):
        self.output_attentions = boolean

        for layer in self.decoder_layers:
            layer.masked_attention.output_attentions = boolean
            layer.encoder_decoder_attention.output_attentions = boolean

    def set_output_hidden_states(self, boolean: bool):
        self.output_hidden_states = boolean


class Transformer(nn.Module, ModuleUtilsMixin, GenerationMixin, ABC):
    def __init__(self,
                 L: int = 3,
                 H: int = 768,
                 A: int = 12,
                 vocab_size: int = 30000,
                 max_sentence_length: int = 512,
                 feed_forward_dim: int = 3072,
                 dropout_rate: float = 0.1,
                 embedding_binding: bool = True,
                 position_embedding_trainable: bool = False,
                 position_embedding_layer=SinusoidalPositionEmbedding,
                 embedding_layer=PositionAwareEmbedding,
                 attention_head=ScaledDotProductAttention,
                 multi_head_attention=MultiHeadAttention,
                 activation_layer=torch.nn.ReLU,
                 feed_forward_layer=FeedForwardLayer,
                 encoder_layer=TransformerEncoderLayer,
                 encoder=TransformerEncoder,
                 decoder_layer=TransformerDecoderLayer,
                 decoder=TransformerDecoder,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False):
        super(Transformer, self).__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        self.encoder = encoder(L, H, A,
                               vocab_size,
                               max_sentence_length,
                               feed_forward_dim,
                               dropout_rate,
                               position_embedding_trainable,
                               position_embedding_layer,
                               embedding_layer,
                               attention_head,
                               multi_head_attention,
                               activation_layer,
                               feed_forward_layer,
                               encoder_layer,
                               output_attentions,
                               output_hidden_states)

        if embedding_binding:
            encoder_embedding_layer = self.encoder.embedding_layer
        else:
            encoder_embedding_layer = None

        self.decoder = decoder(L, H, A,
                               vocab_size,
                               max_sentence_length,
                               feed_forward_dim,
                               dropout_rate,
                               embedding_binding,
                               encoder_embedding_layer,
                               position_embedding_trainable,
                               position_embedding_layer,
                               embedding_layer,
                               attention_head,
                               multi_head_attention,
                               activation_layer,
                               feed_forward_layer,
                               decoder_layer,
                               output_attentions,
                               output_hidden_states)

    def forward(self,
                encoder_input_ids: torch.Tensor,
                decoder_input_ids: torch.Tensor,
                encoder_attention_mask: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.Tensor] = None,
                encoder_token_type_ids: Optional[torch.Tensor] = None,
                encoder_position_ids: Optional[torch.Tensor] = None,
                decoder_token_type_ids: Optional[torch.Tensor] = None,
                decoder_position_ids: Optional[torch.Tensor] = None):
        encoder_output = self.encoder(encoder_input_ids, encoder_attention_mask,
                                      encoder_token_type_ids, encoder_position_ids)

        if not self.output_attentions and not self.output_hidden_states:
            encoded = encoder_output
        else:
            encoded = encoder_output[0]

        decoder_output = self.decoder(decoder_input_ids, encoded, decoder_attention_mask, encoder_attention_mask,
                                      decoder_token_type_ids, decoder_position_ids)

        if not self.output_attentions and not self.output_hidden_states:
            return decoder_output

        output = decoder_output + encoder_output

        return output

    def set_output_attentions(self, boolean: bool):
        self.output_attentions = boolean
        self.encoder.set_output_attentions(boolean)
        self.decoder.set_output_attentions(boolean)

    def set_output_hidden_states(self, boolean: bool):
        self.output_hidden_states = boolean
        self.encoder.set_output_hidden_states(boolean)
        self.decoder.set_output_hidden_states(boolean)


class TransformerEncoderForSequenceClassification(nn.Module, ModuleUtilsMixin):
    def __init__(self, num_labels, **kwargs):
        super(TransformerEncoderForSequenceClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder = TransformerEncoder(**kwargs)
        self.H = self.encoder.H
        self.dropout_rate = self.encoder.dropout_rate
        self.classifier = nn.ModuleList([
            nn.Linear(self.H, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.H),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.H, num_labels)
        ])

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        sequence_output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids)

        if not self.encoder.output_hidden_states and not self.encoder.output_attentions:
            last_hidden_state = sequence_output
        else:
            last_hidden_state = sequence_output[0]

        pooled = last_hidden_state[:, 0, :]
        for layer in self.classifier:
            pooled = layer(pooled)
        logits = pooled

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        hidden_states = None
        attentions = None
        if self.encoder.output_hidden_states:
            hidden_states = sequence_output[1]

            if self.encoder.output_attentions:
                attentions = sequence_output[2]

        else:
            if self.encoder.output_attentions:
                attentions = sequence_output[1]

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def set_output_attentions(self, boolean: bool):
        self.encoder.set_output_attentions(boolean)

    def set_output_hidden_states(self, boolean: bool):
        self.encoder.set_output_hidden_states(boolean)


class TransformerEncoderForTokenClassification(nn.Module, ModuleUtilsMixin):
    def __init__(self, num_labels, **kwargs):
        super(TransformerEncoderForTokenClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder = TransformerEncoder(**kwargs)
        self.H = self.encoder.H
        self.dropout_rate = self.encoder.dropout_rate
        self.classifier = nn.ModuleList([
            nn.Linear(self.H, self.H),
            nn.Tanh(),
            nn.Linear(self.H, self.H),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.H, num_labels)
        ])

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        sequence_output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids)

        if not self.encoder.output_hidden_states and not self.encoder.output_attentions:
            last_hidden_state = sequence_output
        else:
            last_hidden_state = sequence_output[0]

        for layer in self.classifier:
            last_hidden_state = layer(last_hidden_state)
        logits = last_hidden_state

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        hidden_states = None
        attentions = None
        if self.encoder.output_hidden_states:
            hidden_states = sequence_output[1]

            if self.encoder.output_attentions:
                attentions = sequence_output[2]

        else:
            if self.encoder.output_attentions:
                attentions = sequence_output[1]

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def set_output_attentions(self, boolean: bool):
        self.encoder.set_output_attentions(boolean)

    def set_output_hidden_states(self, boolean: bool):
        self.encoder.set_output_hidden_states(boolean)


class TransformerEncoderForCRFClassification(nn.Module, ModuleUtilsMixin):
    def __init__(self, num_labels, **kwargs):
        super(TransformerEncoderForCRFClassification, self).__init__()
        self.num_labels = num_labels
        self.encoder = TransformerEncoder(**kwargs)
        self.H = self.encoder.H
        self.dropout_rate = self.encoder.dropout_rate
        self.projection = nn.ModuleList([
            nn.Linear(self.H, self.H),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.H, num_labels)
        ])
        self.classifier = ConditionalRandomField(num_labels)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                top_k: Optional[int] = None):
        sequence_output = self.encoder(input_ids, attention_mask, token_type_ids, position_ids)

        if not self.encoder.output_hidden_states and not self.encoder.output_attentions:
            last_hidden_state = sequence_output
        else:
            last_hidden_state = sequence_output[0]

        for layer in self.projection:
            last_hidden_state = layer(last_hidden_state)
        logits = self.classifier.viterbi_tags(last_hidden_state, mask=attention_mask, top_k=top_k)

        loss = None
        if labels is not None:
            labels = torch.maximum(labels, torch.zeros_like(labels))
            loss = -self.classifier(last_hidden_state, tags=labels, mask=attention_mask)

        hidden_states = None
        attentions = None
        if self.encoder.output_hidden_states:
            hidden_states = sequence_output[1]

            if self.encoder.output_attentions:
                attentions = sequence_output[2]

        else:
            if self.encoder.output_attentions:
                attentions = sequence_output[1]

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions,
        )

    def set_output_attentions(self, boolean: bool):
        self.encoder.set_output_attentions(boolean)

    def set_output_hidden_states(self, boolean: bool):
        self.encoder.set_output_hidden_states(boolean)
