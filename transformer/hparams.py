# coding=utf-8

class Hyperparams:
    # file path
    source_train = 'de-en/train.tags.de-en.de'
    target_train = 'de-en/train.tags.de-en.en'
    source_test = 'de-en/IWSLT16.TED.tst2014.de-en.de.xml'
    target_test = 'de-en/IWSLT16.TED.tst2014.de-en.en.xml'
    logdir = './model'
    # load data parameter
    min_cnt = 20
    max_len = 10

    # model parameter
    hidden_units = 512
    dropout_rate = 0.3
    num_blocks = 6
    num_heads = 8
    d_ff = 2048
    lr = 0.0001
    num_epochs = 10
    batch_size = 32
