import argparse
import logging
import os
from pprint import pprint

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn.utils import clip_grad_norm
from torchtext import data
from tqdm import tqdm

from models.basic import sequence_cross_entropy
from .data import PTBChar, PTBCharTextField
from .model import PTBModel



log_formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_log_handler = logging.StreamHandler()
console_log_handler.setFormatter(log_formatter)
logger.addHandler(console_log_handler)


def detach_state(state):
    h, c = state
    return h.detach(), c.detach()


def train(args):
    text_field = PTBCharTextField()
    train_dataset, valid_dataset = PTBChar.splits(
        path=args.data, test=None, text_field=text_field)
    text_field.build_vocab(train_dataset)

    train_loader, valid_loader = data.BPTTIterator.splits(
        datasets=(train_dataset, valid_dataset), batch_size=args.batch_size,
        bptt_len=args.bptt_len, device=args.gpu)
    train_loader.repeat = False
    valid_loader.bptt_len = 2000
    valid_loader.batch_size = 20

    model = PTBModel(rnn_type=args.rnn_type,
                     num_chars=len(text_field.vocab),
                     input_size=args.input_size,
                     hidden_size=args.hidden_size,
                     hyper_hidden_size=args.hyper_hidden_size,
                     hyper_embedding_size=args.hyper_embedding_size,
                     use_layer_norm=args.use_layer_norm,
                     dropout_prob=args.dropout_prob)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total parameters: {num_params}')
    if args.gpu > -1:
        model.cuda(args.gpu)
    optimizer = optim.Adam(params=model.parameters())

    summary_writer = SummaryWriter(os.path.join(args.save_dir, 'log', 'train'))

    global_step = 0
    best_valid_loss = 1e10

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        state = hyper_state = None
        for train_batch in tqdm(train_loader, desc=f'Epoch {epoch}: Training'):
            train_inputs = train_batch.text
            train_targets = train_batch.target
            train_logits, state, hyper_state = model(
                inputs=train_inputs, state=state, hyper_state=hyper_state)
            train_loss = sequence_cross_entropy(
                logits=train_logits, targets=train_targets)

            optimizer.zero_grad()
            train_loss.backward()
            clip_grad_norm(parameters=model.parameters(), max_norm=1)
            optimizer.step()

            state = detach_state(state)
            if hyper_state is not None:
                hyper_state = detach_state(hyper_state)
            global_step += 1
            summary_writer.add_scalar(
                tag='train_loss', scalar_value=train_loss.data[0],
                global_step=global_step)

        model.eval()
        valid_loss_sum = valid_loss_denom = 0
        state = hyper_state = None
        for valid_batch in tqdm(valid_loader,
                                desc=f'Epoch {epoch}: Validation'):
            valid_inputs = valid_batch.text
            valid_targets = valid_batch.target
            valid_logits, state, hyper_state = model(
                inputs=valid_inputs, state=state, hyper_state=hyper_state)
            valid_loss = sequence_cross_entropy(
                logits=valid_logits, targets=valid_targets)
            valid_loss_sum += valid_loss.data[0] * valid_inputs.size(0)
            valid_loss_denom += valid_inputs.size(0)
        valid_loss = valid_loss_sum / valid_loss_denom
        valid_bpc = valid_loss / np.log(2)
        summary_writer.add_scalar(
            tag='valid_bpc', scalar_value=valid_bpc, global_step=global_step)
        logging.info(f'Epoch {epoch}: Valid Loss = {valid_loss:.6f}')
        logging.info(f'Epoch {epoch}: Valid BPC = {valid_bpc:.6f}')
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_filename = f'{epoch:03d}-{valid_bpc:.6f}.pt'
            model_path = os.path.join(args.save_dir, model_filename)
            torch.save(model.state_dict(), model_path)
            logging.info('Saved the new best checkpoint')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/ptb_char')
    parser.add_argument('--rnn-type', default='hyperlstm',
                        choices=['hyperlstm', 'lstm'])
    parser.add_argument('--save-dir', required=True)
    parser.add_argument('--input-size', default=49, type=int)
    parser.add_argument('--hidden-size', default=1000, type=int)
    parser.add_argument('--hyper-hidden-size', default=128, type=int)
    parser.add_argument('--hyper-embedding-size', default=4, type=int)
    parser.add_argument('--use-layer-norm', default=False, action='store_true')
    parser.add_argument('--dropout-prob', default=0.1, type=float)
    parser.add_argument('--bptt-len', default=100, type=int)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--gpu', default=-1, type=int)
    parser.add_argument('--max-epoch', default=200, type=int)
    args = parser.parse_args()

    config = {'model': {'rnn_type': args.rnn_type,
                        'input_size': args.input_size,
                        'hidden_size': args.hidden_size,
                        'hyper_hidden_size': args.hyper_hidden_size,
                        'hyper_embedding_size': args.hyper_embedding_size,
                        'use_layer_norm': args.use_layer_norm,
                        'dropout_prob': args.dropout_prob},
              'train': {'bptt_len': args.bptt_len,
                        'batch_size': args.batch_size,}}
    pprint(config)

    os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    file_log_handler = logging.FileHandler(
        os.path.join(args.save_dir, 'train.log'))
    file_log_handler.setFormatter(log_formatter)
    logger.addHandler(file_log_handler)

    train(args)


if __name__ == '__main__':
    main()
