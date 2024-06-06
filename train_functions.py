from tqdm import tqdm
import torch.nn.functional as F
import torch
import random
import numpy as np

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def sst_batch_loss(args, model, batch, optimizer, device):
    b_ids, b_mask, b_labels = (batch['token_ids'],
                               batch['attention_mask'],
                               batch['labels'])

    b_ids = b_ids.to(device)
    b_mask = b_mask.to(device)
    b_labels = b_labels.to(device)

    optimizer.zero_grad()
    logits = model.predict_sentiment(b_ids, b_mask)
    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / args.batch_size
    return loss

def para_batch_loss(args, model, batch, optimizer, device):
    (b_ids1, b_mask1,
     b_ids2, b_mask2,
     b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                  batch['token_ids_2'], batch['attention_mask_2'],
                  batch['labels'])

    b_ids1 = b_ids1.to(device)
    b_mask1 = b_mask1.to(device)
    b_ids2 = b_ids2.to(device)
    b_mask2 = b_mask2.to(device)
    b_labels = b_labels.to(device)
    
    optimizer.zero_grad()
    logits = model.predict_paraphrase(b_ids1, b_mask1, b_ids2, b_mask2)
    loss = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels, reduction='sum') / args.batch_size
    return loss

def sts_batch_loss(args, model, batch, optimizer, device):
    (b_ids1, b_mask1,
     b_ids2, b_mask2,
     b_labels) = (batch['token_ids_1'], batch['attention_mask_1'],
                  batch['token_ids_2'], batch['attention_mask_2'],
                  batch['labels'])

    b_ids1 = b_ids1.to(device)
    b_mask1 = b_mask1.to(device)
    b_ids2 = b_ids2.to(device)
    b_mask2 = b_mask2.to(device)
    b_labels = b_labels.to(device)
    
    optimizer.zero_grad()
    logits = model.predict_similarity(b_ids1, b_mask1, b_ids2, b_mask2)
    loss = F.mse_loss(logits.view(-1), b_labels, reduction='sum') / args.batch_size
    return loss

def training_loop(args, model, optimizer, compute_batch_loss, train_dataloader, dev_dataloader, device, config, eval_fn):
    best_dev_metric = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=args.tqdm_disable):
            loss = compute_batch_loss(args, model, batch, optimizer, device) / args.batch_size

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_metric, *_ = eval_fn(train_dataloader, model, device)
        dev_metric, *_ = eval_fn(dev_dataloader, model, device)

        if dev_metric > best_dev_metric:
            best_dev_metric = dev_metric
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train metric :: {train_metric :.3f}, dev metric :: {dev_metric :.3f}")
