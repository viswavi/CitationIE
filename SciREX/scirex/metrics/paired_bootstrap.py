# Adapted from Graham Neubig's Paired Bootstrap script
# https://github.com/neubig/util-scripts/blob/master/paired-bootstrap.py

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

EVAL_TYPE_ACC = "acc"
EVAL_TYPE_BLEU = "bleu"
EVAL_TYPE_BLEU_DETOK = "bleu_detok"
EVAL_TYPE_PEARSON = "pearson"
EVAL_TYPE_F1 = "f1"
EVAL_TYPE_MACRO_F1 = "macro-f1"
EVAL_TYPE_PREC = "precision"
EVAL_TYPE_REC = "recall"
EVAL_TYPE_AVG = "avg"

EVAL_TYPES = [EVAL_TYPE_ACC,
              EVAL_TYPE_BLEU,
              EVAL_TYPE_BLEU_DETOK,
              EVAL_TYPE_PEARSON,
              EVAL_TYPE_F1,
              EVAL_TYPE_AVG,
              EVAL_TYPE_PREC,
              EVAL_TYPE_REC]

def eval_preproc(data, eval_type='acc'):
  ''' Preprocess into the appropriate format for a particular evaluation type '''
  if type(data) == str:
    data = data.strip()
    if eval_type == EVAL_TYPE_BLEU:
      data = data.split()
    elif eval_type == EVAL_TYPE_PEARSON:
      data = float(data)
    elif eval_type in [EVAL_TYPE_F1, EVAL_TYPE_MACRO_F1, EVAL_TYPE_PREC, EVAL_TYPE_REC]:
      data = float(data)
    elif eval_type == EVAL_TYPE_AVG:
      data = float(data)
  return data

def eval_measure(gold, sys, eval_type='acc'):
  ''' Evaluation measure
  
  This takes in gold labels and system outputs and evaluates their
  accuracy. It currently supports:
  * Accuracy (acc), percentage of labels that match
  * Pearson's correlation coefficient (pearson)
  * BLEU score (bleu)
  * BLEU_detok, on detokenized references and translations, with internal tokenization
  :param gold: the correct labels
  :param sys: the system outputs
  :param eval_type: The type of evaluation to do (acc, pearson, bleu, bleu_detok)
  '''
  if eval_type == EVAL_TYPE_ACC:
    return sum([1 if g == s else 0 for g, s in zip(gold, sys)]) / float(len(gold))
  elif eval_type == EVAL_TYPE_BLEU:
    import nltk
    gold_wrap = [[x] for x in gold]
    return nltk.translate.bleu_score.corpus_bleu(gold_wrap, sys)
  elif eval_type == EVAL_TYPE_PEARSON:
    return np.corrcoef([gold, sys])[0,1]
  elif eval_type == EVAL_TYPE_BLEU_DETOK:
    import sacrebleu
    # make sure score is 0-based instead of 100-based
    return sacrebleu.corpus_bleu(sys, [gold]).score / 100.
  elif eval_type == EVAL_TYPE_F1:
    return f1_score(gold, sys)
  elif eval_type == EVAL_TYPE_MACRO_F1:
    return f1_score(gold, sys, average="macro")
  elif eval_type == EVAL_TYPE_PREC:
    return precision_score(gold, sys)
  elif eval_type == EVAL_TYPE_REC:
    return recall_score(gold, sys)
  elif eval_type == EVAL_TYPE_AVG:
    return np.mean(sys)
  else:
    raise NotImplementedError('Unknown eval type in eval_measure: %s' % eval_type)

def eval_with_paired_bootstrap(gold, sys1, sys2,
                               num_samples=10000, sample_ratio=0.5,
                               eval_type='acc',
                               return_results=False):
  ''' Evaluate with paired boostrap
  This compares two systems, performing a significance tests with
  paired bootstrap resampling to compare the accuracy of the two systems.
  
  :param gold: The correct labels
  :param sys1: The output of system 1
  :param sys2: The output of system 2
  :param num_samples: The number of bootstrap samples to take
  :param sample_ratio: The ratio of samples to take every time
  :param eval_type: The type of evaluation to do (acc, pearson, bleu, bleu_detok)
  '''
  assert(len(gold) == len(sys1))
  assert(len(gold) == len(sys2))
  # Preprocess the data appropriately for they type of eval
  gold = [eval_preproc(x, eval_type) for x in gold]
  sys1 = [eval_preproc(x, eval_type) for x in sys1]
  sys2 = [eval_preproc(x, eval_type) for x in sys2]

  sys1_scores = []
  sys2_scores = []
  wins = [0, 0, 0]
  n = len(gold)
  ids = list(range(n))

  for _ in tqdm(range(num_samples)):
    # Subsample the gold and system outputs
    np.random.shuffle(ids)
    reduced_ids = ids[:int(len(ids)*sample_ratio)]
    reduced_gold = [gold[i] for i in reduced_ids]
    reduced_sys1 = [sys1[i] for i in reduced_ids]
    reduced_sys2 = [sys2[i] for i in reduced_ids]
    # Calculate accuracy on the reduced sample and save stats
    sys1_score = eval_measure(reduced_gold, reduced_sys1, eval_type=eval_type)
    sys2_score = eval_measure(reduced_gold, reduced_sys2, eval_type=eval_type)
    if sys1_score > sys2_score:
      wins[0] += 1
    elif sys1_score < sys2_score:
      wins[1] += 1
    else:
      wins[2] += 1
    sys1_scores.append(sys1_score)
    sys2_scores.append(sys2_score)

  # Print win stats
  wins = [x/float(num_samples) for x in wins]
  print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
  if wins[0] > wins[1]:
    print('(sys1 is superior with p value p=%.10f)\n' % (1-wins[0]))
  elif wins[1] > wins[0]:
    print('(sys2 is superior with p value p=%.10f)\n' % (1-wins[1]))

  # Print system stats
  sys1_scores.sort()
  sys2_scores.sort()
  print('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
  print('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))
  if return_results:
    sys1_summary = (np.mean(sys1_scores), (sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
    sys2_summary = (np.mean(sys2_scores), (sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))
    p_value_lose = 1-wins[0]
    p_value_win = 1-wins[1]
    return sys1_summary, sys2_summary, p_value_lose, p_value_win

def eval_with_hierarchical_paired_bootstrap(gold, sys1_list, sys2_list,
                               num_samples=10000, sample_ratio=0.5,
                               eval_type='acc',
                               return_results=False):
  ''' Evaluate with a hierarchical paired boostrap
  This compares two systems, performing a significance tests with
  paired bootstrap resampling to compare the accuracy of the two systems, with
  two-level sampling: first we sample a model, then we sample data to evaluate
  it on.
  :param gold: The correct labels
  :param sys1: The output of system 1
  :param sys2: The output of system 2
  :param num_samples: The number of bootstrap samples to take
  :param sample_ratio: The ratio of samples to take every time
  :param eval_type: The type of evaluation to do (acc, pearson, bleu, bleu_detok)
  '''
  for sys1 in sys1_list:
    assert(len(gold) == len(sys1))
  for sys2 in sys2_list:
    assert(len(gold) == len(sys2))

  # Preprocess the data appropriately for they type of eval
  gold = [eval_preproc(x, eval_type) for x in gold]
  sys1_list = [[eval_preproc(x, eval_type) for x in sys1] for sys1 in sys1_list]
  sys2_list = [[eval_preproc(x, eval_type) for x in sys2] for sys2 in sys2_list]

  sys1_scores = []
  sys2_scores = []
  wins = [0, 0, 0]
  n = len(gold)
  ids = list(range(n))

  for _ in tqdm(range(num_samples)):
    # Subsample the gold and system outputs
    np.random.shuffle(ids)
    reduced_ids = ids[:int(len(ids)*sample_ratio)]

    sys1_idx = np.random.choice(list(range(len(sys1_list))))
    sys1 = sys1_list[sys1_idx]

    sys2_idx = np.random.choice(list(range(len(sys2_list))))
    sys2 = sys2_list[sys2_idx]

    reduced_gold = [gold[i] for i in reduced_ids]
    reduced_sys1 = [sys1[i] for i in reduced_ids]
    reduced_sys2 = [sys2[i] for i in reduced_ids]
    # Calculate accuracy on the reduced sample and save stats
    sys1_score = eval_measure(reduced_gold, reduced_sys1, eval_type=eval_type)
    sys2_score = eval_measure(reduced_gold, reduced_sys2, eval_type=eval_type)
    if sys1_score > sys2_score:
      wins[0] += 1
    elif sys1_score < sys2_score:
      wins[1] += 1
    else:
      wins[2] += 1
    sys1_scores.append(sys1_score)
    sys2_scores.append(sys2_score)

  # Print win stats
  wins = [x/float(num_samples) for x in wins]
  print('Win ratio: sys1=%.3f, sys2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
  if wins[0] > wins[1]:
    print('(sys1 is superior with p value p=%.10f)\n' % (1-wins[0]))
  elif wins[1] > wins[0]:
    print('(sys2 is superior with p value p=%.10f)\n' % (1-wins[1]))

  # Print system stats
  sys1_scores.sort()
  sys2_scores.sort()
  print('sys1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys1_scores), np.median(sys1_scores), sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
  print('sys2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
          (np.mean(sys2_scores), np.median(sys2_scores), sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))
  if return_results:
    sys1_summary = (np.mean(sys1_scores), (sys1_scores[int(num_samples * 0.025)], sys1_scores[int(num_samples * 0.975)]))
    sys2_summary = (np.mean(sys2_scores), (sys2_scores[int(num_samples * 0.025)], sys2_scores[int(num_samples * 0.975)]))
    p_value_lose = 1-wins[0]
    p_value_win = 1-wins[1]
    return sys1_summary, sys2_summary, p_value_lose, p_value_win
