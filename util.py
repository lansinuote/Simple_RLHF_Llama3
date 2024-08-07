import torch


class TokenizerUtil:

    def __init__(self, checkpoint='google/gemma-2-2b-it'):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained('tokenizer/' + checkpoint)
        
        if checkpoint == 'meta-llama/Meta-Llama-3-8B':
            tokenizer.pad_token = '<|reserved_special_token_0|>'

        self.pad_token_id = tokenizer.pad_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.tokenizer = tokenizer

    def encode(self, sent, max_length=128):
        input_ids = self.tokenizer.encode(sent, add_special_tokens=False)

        input_ids = [self.bos_token_id] + input_ids[:max_length - 2] + [self.eos_token_id]

        input_ids = input_ids + [self.pad_token_id] * (max_length - len(input_ids))

        input_ids = torch.LongTensor(input_ids)
        attention_mask = (input_ids != self.pad_token_id).long()

        return input_ids, attention_mask

    def decode(self, input_ids):
        input_ids = input_ids.tolist()

        if self.eos_token_id in input_ids:
            end = input_ids.index(self.eos_token_id) + 1
            input_ids = input_ids[:end]

        return self.tokenizer.decode(input_ids)

    def pad_to_left(self, input_ids):
        input_ids = input_ids.tolist()
        end = input_ids.index(self.eos_token_id)
        #替换eos为pad
        input_ids[end] = self.pad_token_id
        input_ids = input_ids[end:] + input_ids[:end]

        input_ids = torch.LongTensor(input_ids)
        attention_mask = (input_ids != self.pad_token_id).long()

        return input_ids, attention_mask
    
    
def get_generate(model_actor, input_ids, eos_token_id, pad_token_id, max_length):
    with torch.no_grad():
        #预测最后一个词
        _, logits = model_actor(input_ids=input_ids, attention_mask=(input_ids != pad_token_id).long())

    #直接取最高
    logits = logits[:, -1].argmax(1, keepdim=True)

    #拼合到句子的末尾
    input_ids = torch.cat((input_ids, logits), 1)

    #判断到达最大长度
    if input_ids.shape[1] >= max_length:
        return input_ids

    #判断所有句子都已经结束
    if ((input_ids == eos_token_id).sum(1) >= 1).all():
        return input_ids

    #继续预测下一个词
    return get_generate(model_actor, input_ids, eos_token_id, pad_token_id, max_length)