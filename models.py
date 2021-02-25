import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss



class BertCNNTfIdfForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        config.output_hidden_states = True
        # add cnn params
        self.filter_sizes_ = (3, 4, 5)
        self.num_filters_ = 256
        self.hidden_size_ = config.hidden_size
        self.tfidf = None

        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters_, (k, self.hidden_size_)) for k in self.filter_sizes_])

        self.classifier_0 = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_1 = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            tfidf=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        self.tfidf = tfidf
        # get the sequence output
        last_layer_output = outputs[2][-1] # last_layer_output
        out0 = last_layer_output.unsqueeze(1)
        out0 = torch.cat([self.conv_and_pool(out0, conv) for conv in self.convs], 1)

        assert len(tfidf) == len(last_layer_output)
        batch_layer_out_tfidf = torch.zeros_like(last_layer_output)

        i = 0
        for l_out, tf in zip(last_layer_output, tfidf):
            l_out_tf = torch.mm(tf, l_out)  # (1,hidden_size)
            tf = tf.view(-1, 1)  # (seq_length,1)
            weighted_out = torch.mm(tf, l_out_tf)  # (seq_length, hidden_size)

            batch_layer_out_tfidf[i] = weighted_out
            i += 1

        out1 = torch.as_tensor(batch_layer_out_tfidf)
        out1 = torch.cat([self.conv_and_pool(out1, conv) for conv in self.convs], 1)

        logits0 = self.classifier_0(out0)
        logits1 = self.classifier_1(out1)
        logits = torch.mean(torch.stack([logits0, logits1]), 0)

        # by fc
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = [loss, logits]
        else:
            outputs =[logits, ]

        return outputs  # (loss), logits, (hidden_states), (attentions)
