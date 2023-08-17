import torch
from torch import nn
from models.pytorch_transformers import BertModel, BertConfig, BertTokenizer


class Bert(nn.Module):
    def __init__(self): 
        super(Bert, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained('/opt/data/private/Checkpoints/bert-base-uncased/bert-base-uncased-vocab.txt')
        modelConfig = BertConfig.from_pretrained('/opt/data/private/Checkpoints/bert-base-uncased/bert_config.json')
        self.textExtractor = BertModel.from_pretrained(
            '/opt/data/private/Checkpoints/bert-base-uncased/pytorch_model.bin', config=modelConfig)

    def pre_process2(self, texts):
        tokens, segments, input_masks, text_length = [], [], [], []
        text=texts
        text = '[CLS] ' + text + ' [SEP]'
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        if len(indexed_tokens) > 100:
            indexed_tokens = indexed_tokens[:100]
        for i in range(len(tokens),100):
            tokenized_text.append('[PAD]')
        tokens=tokenized_text
        return tokens
    def pre_process(self, texts):

        tokens, segments, input_masks, text_length = [], [], [], []
        for text in texts:
            text = '[CLS] ' + text + ' [SEP]'
            tokenized_text = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
            if len(indexed_tokens) > 100:
                indexed_tokens = indexed_tokens[:100]
                
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))
        for j in range(len(tokens)):
            padding = [0] * (100 - len(tokens[j]))
            text_length.append(len(tokens[j])+3)
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        tokens = torch.tensor(tokens)
        segments = torch.tensor(segments)
        input_masks = torch.tensor(input_masks)
        text_length = torch.tensor(text_length)

        return tokens, segments, input_masks, text_length


    def forward(self, tokens, segments, input_masks):
        
        output, att_scores = self.textExtractor(tokens, token_type_ids=segments,attention_mask=input_masks)

        return output[0], att_scores


