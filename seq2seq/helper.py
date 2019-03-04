# coding=utf-8

import tqdm

SOURCE_CODES = ['<PAD>', '<UNK>', '<GO>']
TARGET_CODES = ['<PAD>', '<EOS>', '<UNK>', '<GO>']


def load_file(path):
    with open(path, 'r') as file:
        text = file.readlines()
        return text


def load_vocab(path, CODES):
    with open(path, 'r') as file:
        text = file.read()
        vocab = list(set(text.split()))
    token2idx = {token: idx for idx, token in enumerate(CODES + vocab)}
    idx2token = {idx: token for idx, token in enumerate(CODES + vocab)}
    return token2idx, idx2token


# def text2int(sentence, map_dict, max_length=20, is_target=False):
#     text_to_idx = []
#     unk_idx = map_dict.get('<UNK>')
#     pad_idx = map_dict.get('<PAD>')
#     eos_idx = map_dict.get('<EOS>')
#
#     if not is_target:
#         for word in sentence.lower().split():
#             text_to_idx.append(map_dict.get(word, unk_idx))
#     else:
#         for word in sentence.lower().split():
#             text_to_idx.append(map_dict.get(word, unk_idx))
#         text_to_idx.append(eos_idx)
#
#     if len(text_to_idx) > max_length:
#         return text_to_idx[:max_length]
#     else:
#         text_to_idx = text_to_idx + [pad_idx] * (max_length - len(text_to_idx))
#         return text_to_idx
#
#
# if __name__ == '__main__':
#     s_token2idx, s_idx2token = load_vocab('./data/small_vocab_en.txt', SOURCE_CODES)
#     t_token2idx, t_idx2token = load_vocab('./data/small_vocab_fr.txt', TARGET_CODES)
#
#     # 对源句子进行转换 Tx = 20
#     source_text_to_int = []
#     with open('./data/small_vocab_en.txt', 'r') as file:
#         source_text = file.read()
#         lines = source_text.split("\n")
#         lines = lines[1:10]
#         for sentence in tqdm.tqdm(lines):
#             print sentence
#             print text2int(sentence, t_token2idx, 20, is_target=True)
