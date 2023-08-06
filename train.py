import os
import numpy as np
import torch
import sys
import sklearn
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader

eval_path = "./evaluation"
eval_script = './src/conlleval.pl'


if not os.path.isfile(eval_script):
    raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
if not os.path.exists(eval_path):
    os.makedirs(eval_path)


def adjust_learning_rate(optimizer, lr):

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def evaluating(model, datas: DataLoader, best_F: float, use_gpu: bool, id_to_tag: Dict) -> Tuple[float, float, bool]:
    """

    This function will evaluate the model on the given dataset.

    Args:
        model (_type_): 
        datas (DataLoader): the batched dataset to evaluate on
        best_F (float): _description_
        use_gpu (bool): this is a flag to indicate whether to use GPU or not
        id_to_tag (Dict): Dictionary that maps tag ids to tag names

    Returns:
        Tuple[float, float, bool]: (new_F, best_F, save)
    """
    prediction = []
    save = False
    new_F = 0.0
    for batch in datas:
        word_ids, words, tag_ids, tags, seq_len, valid_mask = batch
        if use_gpu:
            val, out = model.inference(word_ids.cuda())
        else:
            val, out = model.inference(word_ids)

        predicted_id = out.cpu()
        
#         The model gives you the predicted ids for the whole sentence, we you only need to compare the valid tokens.
#         extracted the valid tokens from the predicted ids and gold tags.
#         the valid_mask, which is a tensor of the same size as the predicted ids.

        for idx_pred, temp in enumerate(predicted_id):
            predicted_tags_id = torch.masked_select(predicted_id[idx_pred], valid_mask[idx_pred])
            predicted_tags_id = predicted_tags_id.numpy()
            
            predicted_tags = [id_to_tag[tag_id] for tag_id in predicted_tags_id] 
            
            gold_tags_id = torch.masked_select(tag_ids[idx_pred], valid_mask[idx_pred])
            gold_tags_id = gold_tags_id.numpy()
            
            gold_tags = [id_to_tag[tag_id] for tag_id in gold_tags_id] 
            
            for idx_word, temp_1 in enumerate(gold_tags):
          
                prediction.append(str(words[idx_pred][idx_word]) + " " + str(gold_tags[idx_word]) + " " + str(predicted_tags[idx_word]))


    predf = eval_path + '/pred'
    scoref = eval_path + '/score'

    with open(predf, 'w',encoding="utf8") as f:
        f.write('\n'.join(prediction))


    os.system('perl %s < %s > %s' % (eval_script, predf, scoref))

    eval_lines = [l.rstrip() for l in open(scoref, 'r', encoding='utf8')]

    for i, line in enumerate(eval_lines):
        print(line)
        if i == 1:
            new_F = float(line.strip().split()[-1])
            if new_F > best_F:
                best_F = new_F
                save = True
                print('the best F is ', new_F)

    return best_F, new_F, save


def train(model, epochs: int, train_data: DataLoader, dev_data: DataLoader, test_data: DataLoader, use_gpu: bool, id_to_tag: Dict) -> None:
    """

    This function will train the model for the given number of epochs.

    Args:
        model (Any): This is the model to train
        epochs (int): Number of epochs to train for
        train_data (DataLoader): the batched training dataset
        dev_data (DataLoader): the batched development dataset
        test_data (DataLoader): the batched test dataset
        use_gpu (bool): this is a flag to indicate whether to use GPU or not
        id_to_tag (Dict): Dictionary that maps tag ids to tag names

    Returns:
        None
    """
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    loss = 0.0
    best_dev_F = -1.0
    best_test_F = -1.0
    all_F = [[0, 0]]
    plot_every = 20
    eval_every = 20
    count = 0
    sys.stdout.flush()

    model.train(True)

    for epoch in range(1, epochs):
        for iter, batch in tqdm(enumerate(train_data)):
            word_ids, word, tag_ids, tag, seq_len, mask_list = batch
            model.zero_grad()
            count += 1
#             print("count :", count)
            if use_gpu:
                neg_log_likelihood = model(
                    word_ids.cuda(),
                    tag_ids.cuda(),
                )
            else:
                neg_log_likelihood = model(word_ids, tag_ids)
            loss += neg_log_likelihood.data.item()
            neg_log_likelihood.backward()
            optimizer.step()

            if count % plot_every == 0:
                loss /= plot_every
                print(loss)

            if ((count % (eval_every) == 0 and count > (eval_every * 20)) or (count % (eval_every * 4) == 0 and count <
                    (eval_every * 20))):
                print("Inside if condition!!")
                model.train(False)
                best_test_F, new_test_F, _ = evaluating(
                    model, test_data, best_test_F, use_gpu, id_to_tag)
                best_dev_F, new_dev_F, save = evaluating(
                    model, dev_data, best_dev_F, use_gpu, id_to_tag)
                if save:
                    torch.save(model, 'model.pt')
                sys.stdout.flush()

                all_F.append([new_dev_F, new_test_F])
                model.train(True)

            if count % len(train_data) == 0:
                adjust_learning_rate(
                    optimizer, lr=learning_rate / (1 + 0.05 * count / len(train_data)))
                
            losses.append(loss)

    plt.plot(losses)
    plt.show()
