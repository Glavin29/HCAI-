import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import BertModel


class GradReverse(Function):
    """
    Gradient Reversal Layer.
    Forward pass acts as identity.
    Backward pass reverses the gradient multiplied by lambda.
    """

    @staticmethod
    def forward(ctx, x, lambda_=1.0):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class FairBERTModel(nn.Module):
    """
    BERT-based resume classifier with an adversarial gender classifier.
    The main classifier predicts resume category.
    The adversarial classifier tries to predict gender from the same representation.
    Gradient reversal discourages the BERT representation from storing gender information.
    """

    def __init__(self, num_labels):
        super(FairBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, num_labels)
        self.gender_adv = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, lambda_adv=0.1):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output

        class_logits = self.classifier(pooled)

        reversed_pooled = GradReverse.apply(pooled, lambda_adv)
        gender_logits = self.gender_adv(reversed_pooled)

        return class_logits, gender_logits