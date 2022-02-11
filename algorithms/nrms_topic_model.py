import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BaseConfig():
    """
    General configurations appiled to all models
    """
    BCELoss = True
    regularize_entropy = False
    entropy_alpha = 0.5
    num_epochs = 10
    num_batches_show_loss = 50  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 1000
    batch_size = 400
    learning_rate = 3e-4
    num_workers = 4  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 1
    entity_freq_threshold = 2
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 1  # K
    dropout_probability = 0.2
    # Modify the following by the output of `src/dataprocess.py`
    num_words = 1 + 101220
    num_categories = 1 + 285
    num_entities = 1 + 21842
    num_users = 1 + 711222
    word_embedding_dim = 300
    reduction_dim = 64
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200


class NRMSConfig(BaseConfig):
    dataset_attributes = {"news": ['subcategory', 'title'], 
                          "record": []
    }
    # For multi-head self-attention
    num_attention_heads = 15
    
config = NRMSConfig()


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask=None):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads, device):
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, Q, K=None, V=None, length=None):
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(batch_size, -1, self.num_attention_heads,
                               self.d_k).transpose(1, 2)
        v_s = self.W_V(V).view(batch_size, -1, self.num_attention_heads,
                               self.d_v).transpose(1, 2)

        if length is not None:
            maxlen = Q.size(1)
            attn_mask = torch.arange(maxlen).to(self.device).expand(
                batch_size, maxlen) < length.to(self.device).view(-1, 1)
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, maxlen,
                                                      maxlen)
            attn_mask = attn_mask.unsqueeze(1).repeat(1,
                                                      self.num_attention_heads,
                                                      1, 1)
        else:
            attn_mask = None

        context, attn = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s,
                                                            attn_mask)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.num_attention_heads * self.d_v)
        return context
    
class AdditiveAttention(torch.nn.Module):
    """
    A general additive attention module.
    Originally for NAML.
    """
    def __init__(self,
                 query_vector_dim,
                 candidate_vector_dim,
                 writer=None,
                 tag=None,
                 names=None):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(candidate_vector_dim, query_vector_dim)
        self.attention_query_vector = nn.Parameter(
            torch.empty(query_vector_dim).uniform_(-0.1, 0.1))
        # For tensorboard
        self.writer = writer
        self.tag = tag
        self.names = names
        self.local_step = 1

    def forward(self, candidate_vector):
        """
        Args:
            candidate_vector: batch_size, candidate_size, candidate_vector_dim
        Returns:
            (shape) batch_size, candidate_vector_dim
        """
        # batch_size, candidate_size, query_vector_dim
        temp = torch.tanh(self.linear(candidate_vector))
        # batch_size, candidate_size
        candidate_weights = F.softmax(torch.matmul(
            temp, self.attention_query_vector),
                                      dim=1)
        if self.writer is not None:
            assert candidate_weights.size(1) == len(self.names)
            if self.local_step % 10 == 0:
                self.writer.add_scalars(
                    self.tag, {
                        x: y
                        for x, y in zip(self.names,
                                        candidate_weights.mean(dim=0))
                    }, self.local_step)
            self.local_step += 1
        # batch_size, candidate_vector_dim
        target = torch.bmm(candidate_weights.unsqueeze(dim=1),
                           candidate_vector).squeeze(dim=1)
        return target

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding, device):
        super(NewsEncoder, self).__init__()
        self.device = device
        self.config = config
        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)

        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads, device)
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)

    def forward(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_words_title, word_embedding_dim
        idx = news["title"]
        if idx.dim() == 2:
            idx = idx.unsqueeze(0)
        news_vector = F.dropout(self.word_embedding(idx.to(self.device)),
                                p=self.config.dropout_probability,
                                training=self.training)
        # batch_size, num_words_title, word_embedding_dim
        multihead_news_vector = self.multihead_self_attention(news_vector)
        multihead_news_vector = F.dropout(multihead_news_vector,
                                          p=self.config.dropout_probability,
                                          training=self.training)
        # batch_size, word_embedding_dim
        final_news_vector = self.additive_attention(multihead_news_vector)
        return final_news_vector

class UserEncoder(torch.nn.Module):
    def __init__(self, config, device):
        super(UserEncoder, self).__init__()
        self.config = config
        self.multihead_self_attention = MultiHeadSelfAttention(
            config.word_embedding_dim, config.num_attention_heads, device)
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.word_embedding_dim)
        self.dimmension_reduction = nn.Sequential(nn.Linear(config.word_embedding_dim, config.word_embedding_dim),
                                                  nn.Dropout(config.dropout_probability),
                                                  nn.ReLU(),
                                                  nn.Linear(config.word_embedding_dim, config.reduction_dim))
    def forward(self, user_vector):
        """
        Args:
            user_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        multihead_user_vector = self.multihead_self_attention(user_vector)
        # batch_size, reduction_dim
        final_user_vector = self.dimmension_reduction(self.additive_attention(multihead_user_vector))
        return final_user_vector

class TopicEncoder(torch.nn.Module):
    def __init__(self, config, device):
        super(TopicEncoder, self).__init__()
        self.device = device
        self.config = config
        self.word_embedding = nn.Embedding(config.num_categories,
                                           config.reduction_dim,
                                           padding_idx=0)
        self.mlp_head = nn.Sequential(nn.Linear(config.reduction_dim, config.reduction_dim),
                                      nn.Dropout(p=self.config.dropout_probability),
                                      nn.ReLU(),
                                      nn.Linear(config.reduction_dim, config.reduction_dim))
    def forward(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                }
        Returns:
            (shape) batch_size, word_embedding_dim
        """
        # batch_size, reduction_dim
        news_vector = F.dropout(self.word_embedding(news["subcategory"].to(self.evice)),
                                p=self.config.dropout_probability,
                                training=self.training)
        # batch_size, reduction_dim
        final_news_vector = self.mlp_head(news_vector)
        return final_news_vector
    
    def get_all_topic_embeddings(self, topic_order):
        # topic_indices = torch.LongTensor(range(1, self.config.num_categories))
        topic_indices = torch.LongTensor(topic_order)
        topic_vector = F.dropout(self.word_embedding(topic_indices.to(self.device)),
                                p=self.config.dropout_probability,
                                training=True)
        final_topic_vector = self.mlp_head(topic_vector)
        return final_topic_vector # num_topic, reduction_dim
    
class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        """
        Args:
            candidate_news_vector: batch_size, candidate_size, X
            user_vector: batch_size, X
        Returns:
            (shape): batch_size
        """
        # batch_size, candidate_size
        probability = torch.bmm(candidate_news_vector,
                                user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        return probability
    
class NRMS(torch.nn.Module):
    """
    NRMS network.
    Input 1 + K candidate news and a list of user clicked news, produce the click probability.
    """
    def __init__(self,device, pretrained_word_embedding=None):
        super(NRMS, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding, device)
        self.user_encoder = UserEncoder(config, device)
        self.topic_encoder = TopicEncoder(config, device)
        self.click_predictor = DotProductClickPredictor()
        self.all_topic_vector = None

    def forward(self, candidate_news, clicked_news):
        """
        Args:
            candidate_news:
                [
                    {
                        "title": batch_size * num_words_title
                    } * (1 + K)
                ]
            clicked_news:
                [
                    {
                        "title":batch_size * num_words_title
                    } * num_clicked_news_a_user
                ]
        Returns:
          click_probability: batch_size, 1 + K
        """
        # batch_size, 1 + K, reduction_dim
        candidate_news_vector = torch.stack(
            [self.topic_encoder(x) for x in candidate_news], dim=1)
        # batch_size, num_clicked_news_a_user, word_embedding_dim
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        # batch_size, reduction_dim
        user_vector = self.user_encoder(clicked_news_vector)
        # batch_size, 1 + K
        click_probability = self.click_predictor(candidate_news_vector,
                                                 user_vector)
        return click_probability

    def get_news_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, reduction_dim
        """
        # batch_size, reduction_dim
        return self.news_encoder(news)
    
    def get_topic_vector(self, news):
        """
        Args:
            news:
                {
                    "title": batch_size * num_words_title
                },
        Returns:
            (shape) batch_size, reduction_dim
        """
        # batch_size, reduction_dim
        return self.topic_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        """
        Args:
            clicked_news_vector: batch_size, num_clicked_news_a_user, word_embedding_dim
        Returns:
            (shape) batch_size, reduction_dim
        """
        # batch_size, word_embedding_dim
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        """
        Args:
            news_vector: candidate_size, reduction_dim
            user_vector: reduction_dim
        Returns:
            click_probability: candidate_size
        """
        # candidate_size
        return self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
        
    def get_all_topic_score(self, user_vector, topic_order):
        """
        Args:
            user_vector: reduction_dim
        Returns:
            topic_probability: candidate_size
        """

        all_topic_vector = self.topic_encoder.get_all_topic_embeddings(topic_order) # num_topic, reduction_dim

        return self.click_predictor(
            all_topic_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0) # num_topic
