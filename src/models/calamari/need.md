# experiment with the other architetcures of hte calamari
# no need for cross fold 
# could think about ensembles (deocder support), maybe could combined pretrianed and fine tuned model

#! ema weights


On their data, 5-model vote cut CER from 0.155% → 0.114% (UW3) and 9.1% → 7.1% (incunable, 50 lines). Reul DAS 2018: 16–62% of remaining errors removed, plus another 5–10% from using confidences. That is the lever you have not used. This repo also has no voter and only greedy CTC.


Calamari does not fuse architectures into one network. At inference it runs several independent models and confidence-votes their CTC probabilities (not majority vote of strings). Diversity comes from different folds, different CNN–LSTM graphs, or different pretrained bases. Cross-fold is optional; mixing a pretrained voter with a fine-tuned one (as in your need.md) is what Reul et al. recommend.

On their data, 5-model vote cut CER from 0.155% → 0.114% (UW3) and 9.1% → 7.1% (incunable, 50 lines). Reul DAS 2018: 16–62% of remaining errors removed, plus another 5–10% from using confidences. That is the lever you have not used. This repo also has no voter and only greedy CTC.

Your graph is already deeper (3 LSTMs) than the 2018 default (1 LSTM), but the CNN is thinner (40/60 vs 64/128 filters). A second voter with a different graph is more useful than a fourth LSTM on the same one.


increasing the cnn 



What you are missing (not more params)
Confidence vote 3–5 existing checkpoints (LSTM 2 vs 3, pretrained vs fine-tuned, two seeds).
CTC beam / word-beam + a small Classical Armenian lexicon (AUA: 3.6% → 1.7% from the decoder alone).
Per-script or per-book fine-tune (erkatagir / bolorgir / notrgir / shlagir mixed in one model is a known killer).
Abbreviation-aware GT (Calfa expands contractions; up to 90% of words can be abbreviated).
Your own val already showed 44% CER on 1–9 character lines vs ~10% on mid-length lines — that is data/GT, not capacity.