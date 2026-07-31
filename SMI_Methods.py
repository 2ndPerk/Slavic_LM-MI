#edit because git is being weird
#imports also need to be here for some reason
import pandas as pd
from tqdm.notebook import tqdm
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import CrossEntropyLoss
import numpy as np

from os import chdir
from pathlib import Path


def make_paths_relative_to_root():
    """Always use the same, absolute (relative to root) paths

    which makes moving the notebooks around easier.
    """
    top_level = Path(__file__).parent

    chdir(top_level)


make_paths_relative_to_root()

#Data
def prep_data(source, words):
    source = "test/"+source+".txt"
    file = open(source, "r", encoding = 'utf-8')
    lines = file.readlines()

    data = [[],[],[],[]]
    data_3line = [[],[],[],[]]
    data_3line_unfilled = [[],[],[],[]]
    for n in range(4):
        for l in range(12):
            line = lines[(12*n) + l]
            if line != "\n":
                data[n].append(line)

                prev = lines[l-1].replace("{}", words[n][l-1]) if l > 0 else ""
                next = lines[l+1].replace("{}", words[n][l+1]) if l < 11 else ""

                data_3line[n].append(prev + line + next)

                prev = lines[l-1].replace("{}", '_') if l > 0 else ""
                next = lines[l+1].replace("{}", '_') if l < 11 else ""

                data_3line_unfilled[n].append(prev + line + next)
        
    return [data, data_3line, data_3line_unfilled]

def prep_words(source):
    source = "test/"+source+"_Words.txt"
    file = open(source, "r", encoding = 'utf-8')
    lines = file.readlines()
    
    words = [[],[],[],[]]
    for n in range(4):
        for l in range(12):
            line = lines[(n*12)+l]
            words[n].append(line)
            
    return words

# source: https://github.com/XuhuiZhou/CATS
def uni_predict(text, model, tokenizer):
    # Tokenized input
    # text = "[CLS] I got restricted because Tom reported my reply [SEP]"
    text = text
    tokenized_text = tokenizer.tokenize(text)
    sentence_score = 0
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    length = len(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor = tokens_tensor.to('cuda')
    #masked_tensor = torch.tensor([masked_index])
    with torch.no_grad():
        outputs = model(tokens_tensor, labels= tokens_tensor)
    loss = outputs[0]
    sentence_score = -loss
    return sentence_score

def greedy_select(df):
    selected_positions = []
    remaining_rows = set(df.index)
    remaining_columns = set(df.columns)

    while len(remaining_rows) > 0 and len(remaining_columns) > 0:
        min_value = float('inf')
        min_position = None

        # Find the smallest value and its position
        for row in remaining_rows:
            for column in remaining_columns:
                value = df.at[row, column]
                if value < min_value:
                    min_value = value
                    min_position = (row, column)

        # Remove the row and column
        remaining_rows.remove(min_position[0])
        remaining_columns.remove(min_position[1])

        # Add the position to the selected list
        selected_positions.append(min_position)

    return sorted(selected_positions, key=lambda x: x[0])

def score_model(model, tokenizer, data, opts):
    r_scores = []
    df_data = []
    t1_score = 0
    t3_score = 0
    for d in tqdm(data):
        correct = opts[data.index(d)]
        scores = {}
        for o in opts:
            sentence = d.replace("{}", o)
            scores[o] = float(uni_predict(sentence, model, tokenizer).item())
        df_data.append(scores)
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        i = 0
        for key, value in scores:
            t1_score += ((i == 0) and (key == correct))
            t3_score += ((i < 3) and (key == correct))
            i += 1

    df = pd.DataFrame(df_data)
    r_scores.append(t1_score / 12)
    r_scores.append(t3_score / 12)

    # Normalize each row
    df = df.div(df.mean(axis=1), axis=0)

    # Hungarian Algorithm for linear sum assignment optimizes score over all selections
    x, y = linear_sum_assignment(df)
    out = pd.DataFrame({'Word': df.columns[y], 'Sentence': df.index[x]})
    final_score = sum(opts[n] == out.iloc[n]['Word'] for n in range(12)) / 12
    r_scores.append(final_score)

    # Greedy Selection tries to maximize high confidence picks instead of overall score
    greedy_values = greedy_select(df)
    final_score = sum(opts[n] == greedy_values[n][1] for n in range(12)) / 12
    r_scores.append(final_score)
    return r_scores


# source: https://github.com/XuhuiZhou/CATS
#For BERT model testing
def bert_predict(text, model, tokenizer):
    # Tokenized input
    # text = "[CLS] I got restricted because Tom reported my reply [SEP]"
    text = "[CLS] " + text + " [SEP]" #special token for BERT, RoBERTa
    tokenized_text = tokenizer.tokenize(text)
    sentence_score = 0
    length = len(tokenized_text)-2
    for masked_index in range(1,len(tokenized_text)-1):
        # Mask a token that we will try to predict back with `BertForMaskedLM`
        masked_word = tokenized_text[masked_index]
        #tokenized_text[masked_index] = '<mask>' #special token for XLNet
        tokenized_text[masked_index] = '[MASK]' #special token for BERT, RoBerta
        # Convert token to vocabulary indices
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        index = torch.tensor(tokenizer.convert_tokens_to_ids(masked_word))
        tokens_tensor = torch.tensor([indexed_tokens])
        tokens_tensor = tokens_tensor.to('cuda')
        index = index.to('cuda')
        #masked_tensor = torch.tensor([masked_index])
        with torch.no_grad():
            outputs = model(tokens_tensor)
        prediction_scores = outputs[0]
        prediction_scores = prediction_scores.view(-1, model.config.vocab_size)
        prediction_scores = prediction_scores[masked_index].unsqueeze(0)
        loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
        masked_lm_loss = loss_fct(prediction_scores, index.view(-1))
        tokenized_text[masked_index] = masked_word
        sentence_score -= masked_lm_loss.item()
        tokenized_text[masked_index] = masked_word
    sentence_score = sentence_score/length
    return sentence_score

def score_model_bert(model, tokenizer, data, opts): 
    r_scores = []
    df_data = []
    t1_score = 0
    t3_score = 0
    for d in tqdm(data):
        correct = opts[data.index(d)]
        scores = {}
        for o in opts:
            sentence = d.replace("{}", o)
            scores[o] = bert_predict(sentence, model, tokenizer)
        df_data.append(scores)
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        i = 0
        for key, value in scores:
            t1_score += ((i == 0) and (key == correct))
            t3_score += ((i < 3) and (key == correct))
            i += 1
            
    df = pd.DataFrame(df_data)
    r_scores.append(t1_score/12)
    r_scores.append(t3_score/12)
    
    #normalize each row
    df = df.div(df.mean(axis=1), axis=0)
    
    #Hungarian Algorithm for linear sum assignment optimizes score over all selections
    x, y = linear_sum_assignment(df)
    out = pd.DataFrame({'Word': df.columns[y], 'Sentence': df.index[x]})
    final_score = sum(opts[n] == out.iloc[n]['Word'] for n in range(12)) / 12
    r_scores.append(final_score)
    
    #Greedy Selection tries to maximize high confidence picks instead of overall score
    greedy_values = greedy_select(df)
    final_score = sum(opts[n] == greedy_values[n][1] for n in range(12)) / 12
    r_scores.append(final_score)
    return r_scores