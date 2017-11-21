import argparse
from pprint import pprint

import numpy as np
import torch
import yaml
from torchtext import data
from tqdm import tqdm

from models.basic import sequence_cross_entropy
from .data import PTBChar, PTBCharTextField
from .model import PTBModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/ptb_char')
    parser.add_argument('--model', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--gpu', default=-1, type=int)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f)
    pprint(config)

    text_field = PTBCharTextField()
    train_dataset, test_dataset = PTBChar.splits(
        path=args.data, validation=None, text_field=text_field)
    text_field.build_vocab(train_dataset)

    test_loader = data.BPTTIterator(
        dataset=test_dataset, batch_size=1, bptt_len=100, train=False,
        device=args.gpu)

    model = PTBModel(num_chars=len(text_field.vocab), **config['model'])
    model.load_state_dict(torch.load(args.model))
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {num_params}')

    if args.gpu > -1:
        model.cuda(args.gpu)

    model.eval()

    state = hyper_state = None
    test_bpc_sum = test_bpc_denom = 0
    for test_batch in tqdm(test_loader):
        test_inputs = test_batch.text
        test_targets = test_batch.target
        test_logits, state, hyper_state = model(
            inputs=test_inputs, state=state, hyper_state=hyper_state)
        test_loss = sequence_cross_entropy(
            logits=test_logits, targets=test_targets)
        test_bpc_sum += (test_loss.data[0] / np.log(2)) * test_inputs.size(0)
        test_bpc_denom += test_inputs.size(0)
    test_bpc = test_bpc_sum / test_bpc_denom

    print(f'Test BPC = {test_bpc:.6f}')



if __name__ == '__main__':
    main()
