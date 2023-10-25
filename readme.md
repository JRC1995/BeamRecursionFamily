### Official Repository For:

#### 1. Efficient Beam Tree Recursion - Jishnu Ray Chowdhury, Cornelia Caragea NeurIPS 2023
#### 2. Recursion in Recursion: Two-Level Nested Recursion for Length Generalization with Scalability - Jishnu Ray Chowdhury, Cornelia Caragea NeurIPS 2023

### Credits:
* CRvNN-related models are adapted from: https://github.com/JRC1995/Continuous-RvNN
* ```models/encoder/OrderedMemory.py``` adapted from: https://github.com/yikangshen/Ordered-Memory
* BT-GRC, and GT-GRC related models are adapted from: https://github.com/jihunchoi/unsupervised-treelstm
* ```optimizers/ranger.py``` adapted from: https://github.com/anoidgit/transformer/blob/master/optm/ranger.py
* ```extensions/, models/modules/S4.py``` is copied from: https://github.com/HazyResearch/state-spaces/tree/02e1ba8731ceea90db8343d6e9e150509607bec7
* ```preprocess/IMBD_lra.py``` is adapted from https://github.com/google-research/long-range-arena
* ```models/encoders/MEGA.py``` and ```fairseq/``` adapted from https://github.com/facebookresearch/mega

### Requirements
* pytorch                      1.10.0
* pytorch-lightning            1.9.3
* tqdm                         4.62.3
* tensorflow-datasets          4.5.2
* typing_extensions            4.5.0
* pykeops                      2.1.1
* jsonlines                    2.0.0
* einops                       0.6.0
* torchtext                    0.8.1

### Data Setup
* Put the Logical Inference data files (train0,train1,train2,.....test12) (downloaded from https://github.com/yikangshen/Ordered-Memory/tree/master/data/propositionallogic) in data/proplogic/
* Download the ListOps data (along with extrapolation data) from the urls here: https://github.com/facebookresearch/latent-treelstm/blob/master/data/listops/external/urls.txt and put the tsv files in data/listops/
* Run all the make*.py files in data/listops/ to create relevant splits (exact splits used in the paper will be released later) 
* Download LRA (https://github.com/google-research/long-range-arena) dataset
* From LRA dataset put the ListOps basic_test.tsv (LRA test set) in data/listops
* From LRA dataset put the ListOps basic_train.tsv, basic_val.tsv, and basic_test.tsv in data/listops_lra
* From LRA dataset's Retrieval task put new_aan_pairs.train.tsv, new_aan_pairs.eval.tsv, and new_aan_pairs.test.tsv in data/AAN.
* Download IMDB original split from here: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz, extract and put the acllmdb folder in data/IMDB/ 
* Download IMDB contrast set from here: https://github.com/allenai/contrast-sets/tree/main/IMDb/data, put the dev_original.tsv, dev_contrast.tsv and test_contrast.tsv in data/IMDB/
* Download IMDB counterfactual test set from here: https://github.com/acmi-lab/counterfactually-augmented-data/blob/master/sentiment/new/test.tsv; rename it to "test_counterfactual.tsv". Put it in data/IMDB
* Download MNLI datasets (https://cims.nyu.edu/~sbowman/multinli/) and put them in data/MNLI/
* Download MNLI stress tests (https://abhilasharavichander.github.io/NLI_StressTest/) and put them in data/MNLI/
* Download Conjunctive NLI dev set from here (https://github.com/swarnaHub/ConjNLI/tree/master/data/NLI)  and put it (conj_dev.tsv) in data/MNLI/
* Put the SNLI files (downloaded and extracted from here: https://nlp.stanford.edu/projects/snli/) in data/SNLI/  
* Download https://nlp.stanford.edu/projects/snli/snli_1.0_test_hard.jsonl and put it in data/SNLI/
* Extract dataset.jsonl from here: https://github.com/BIU-NLP/Breaking_NLI and put it in data/SNLI/
* Download the test.tsv from here: https://github.com/acmi-lab/counterfactually-augmented-data/tree/master/NLI/revised_combined and put it in data/SNLI/revised_combined/  
* Download QQP (quora_duplicate_questions.tsv) from here: https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs, and put it in data/QQP/
* Follow instructions from here (https://github.com/google-research-datasets/paws) to generate PAWS QQP dev_and_test.tsv and put it in data/QQP/PAWS_QQP/
* Follow instructions from here (https://github.com/google-research-datasets/paws) to generate PAWS WIKI test.tsv and put it in data/QQP/PAWS_WIKI/
* Put glove.840B.300d.txt (download glove.840B.300d.zip from here:https://nlp.stanford.edu/projects/glove/) in embeddings/glove/

### Processing Data
* Go to preprocess/ and run each preprocess files to preprocess the corresponding data (process_SNLI_addon.py must be run after process_SNLI.py; otherwise no order requirement)


You can verify if the data is properly set up from the tree below.
```
├───data
│   ├───AAN
│   │       new_aan_pairs.eval.tsv
│   │       new_aan_pairs.test.tsv
│   │       new_aan_pairs.train.tsv
│   │
│   ├───IMDB
│   │   │   dev_contrast.tsv
│   │   │   dev_original.tsv
│   │   │   test_contrast.tsv
│   │   │   test_counterfactual.tsv
│   │   │
│   │   └───aclImdb
│   │       │   imdb.vocab
│   │       │   imdbEr.txt
│   │       │   README
│   │       │
│   │       ├───test
│   │       │   │   labeledBow.feat
│   │       │   │   urls_neg.txt
│   │       │   │   urls_pos.txt
│   │       │   │
│   │       │   ├───neg
│   │       │   └───pos
│   │       └───train
│   │           │   labeledBow.feat
│   │           │   unsupBow.feat
│   │           │   urls_neg.txt
│   │           │   urls_pos.txt
│   │           │   urls_unsup.txt
│   │           │
│   │           ├───neg
│   │           ├───pos
│   │           └───unsup
│   ├───listops
│   │       base.py
│   │       basic_test.tsv
│   │       dev_d7s.tsv
│   │       load_listops_data.py
│   │       make_depth_dev_data.py
│   │       make_depth_ndr_data.py
│   │       make_depth_test_data.py
│   │       make_depth_train_data.py
│   │       make_depth_train_data_extra.py
│   │       make_iid_data.py
│   │       make_odd_25depth_data.py
│   │       make_ood_10arg_data.py
│   │       make_ood_15arg_data.py
│   │       test_200_300.tsv
│   │       test_300_400.tsv
│   │       test_400_500.tsv
│   │       test_500_600.tsv
│   │       test_600_700.tsv
│   │       test_700_800.tsv
│   │       test_800_900.tsv
│   │       test_900_1000.tsv
│   │       test_d20s.tsv
│   │       test_dg8s.tsv
│   │       test_iid_arg.tsv
│   │       test_ood_10arg.tsv
│   │       test_ood_15arg.tsv
│   │       train_d20s.tsv
│   │       train_d6s.tsv
│   │       __init__.py
│   │
│   ├───listops_lra
│   │       basic_test.tsv
│   │       basic_train.tsv
│   │       basic_val.tsv
│   │
│   ├───MNLI
│   │   │   conj_dev.tsv
│   │   │   multinli_0.9_test_matched_unlabeled.jsonl
│   │   │   multinli_0.9_test_matched_unlabeled_hard.jsonl
│   │   │   multinli_0.9_test_mismatched_unlabeled.jsonl
│   │   │   multinli_0.9_test_mismatched_unlabeled.jsonl.zip
│   │   │   multinli_0.9_test_mismatched_unlabeled_hard.jsonl
│   │   │   multinli_1.0_dev_matched.jsonl
│   │   │   multinli_1.0_dev_matched.txt
│   │   │   multinli_1.0_dev_mismatched.jsonl
│   │   │   multinli_1.0_dev_mismatched.txt
│   │   │   multinli_1.0_train.jsonl
│   │   │   multinli_1.0_train.txt
│   │   │   paper.pdf
│   │   │   README.txt
│   │   │
│   │   ├───Antonym
│   │   │       multinli_0.9_antonym_matched.jsonl
│   │   │       multinli_0.9_antonym_matched.txt
│   │   │       multinli_0.9_antonym_mismatched.jsonl
│   │   │       multinli_0.9_antonym_mismatched.txt
│   │   │
│   │   ├───Length_Mismatch
│   │   │       multinli_0.9_length_mismatch_matched.jsonl
│   │   │       multinli_0.9_length_mismatch_matched.txt
│   │   │       multinli_0.9_length_mismatch_mismatched.jsonl
│   │   │       multinli_0.9_length_mismatch_mismatched.txt
│   │   │
│   │   ├───Negation
│   │   │       multinli_0.9_negation_matched.jsonl
│   │   │       multinli_0.9_negation_matched.txt
│   │   │       multinli_0.9_negation_mismatched.jsonl
│   │   │       multinli_0.9_negation_mismatched.txt
│   │   │
│   │   ├───Numerical_Reasoning
│   │   │       .DS_Store
│   │   │       multinli_0.9_quant_hard.jsonl
│   │   │       multinli_0.9_quant_hard.txt
│   │   │
│   │   ├───Spelling_Error
│   │   │       multinli_0.9_dev_gram_contentword_swap_perturbed_matched.jsonl
│   │   │       multinli_0.9_dev_gram_contentword_swap_perturbed_matched.txt
│   │   │       multinli_0.9_dev_gram_contentword_swap_perturbed_mismatched.jsonl
│   │   │       multinli_0.9_dev_gram_contentword_swap_perturbed_mismatched.txt
│   │   │       multinli_0.9_dev_gram_functionword_swap_perturbed_matched.jsonl
│   │   │       multinli_0.9_dev_gram_functionword_swap_perturbed_matched.txt
│   │   │       multinli_0.9_dev_gram_functionword_swap_perturbed_mismatched.jsonl
│   │   │       multinli_0.9_dev_gram_functionword_swap_perturbed_mismatched.txt
│   │   │       multinli_0.9_dev_gram_keyboard_matched.jsonl
│   │   │       multinli_0.9_dev_gram_keyboard_matched.txt
│   │   │       multinli_0.9_dev_gram_keyboard_mismatched.jsonl
│   │   │       multinli_0.9_dev_gram_keyboard_mismatched.txt
│   │   │       multinli_0.9_dev_gram_swap_matched.jsonl
│   │   │       multinli_0.9_dev_gram_swap_matched.txt
│   │   │       multinli_0.9_dev_gram_swap_mismatched.jsonl
│   │   │       multinli_0.9_dev_gram_swap_mismatched.txt
│   │   │
│   │   └───Word_Overlap
│   │           multinli_0.9_taut2_matched.jsonl
│   │           multinli_0.9_taut2_matched.txt
│   │           multinli_0.9_taut2_mismatched.jsonl
│   │           multinli_0.9_taut2_mismatched.txt
│   │
│   ├───proplogic
│   │       generate_neg_set_data.py
│   │       test0
│   │       test1
│   │       test10
│   │       test11
│   │       test12
│   │       test2
│   │       test3
│   │       test4
│   │       test5
│   │       test6
│   │       test7
│   │       test8
│   │       test9
│   │       train0
│   │       train1
│   │       train10
│   │       train11
│   │       train12
│   │       train2
│   │       train3
│   │       train4
│   │       train5
│   │       train6
│   │       train7
│   │       train8
│   │       train9
│   │       __init__.py
│   │
│   ├───QQP
│   │   │   quora_duplicate_questions.tsv
│   │   │
│   │   ├───PAWS_QQP
│   │   │       dev_and_test.tsv
│   │   │
│   │   └───PAWS_WIKI
│   │           test.tsv
│   │
│   └───SNLI
│       │   dataset.jsonl
│       │   README.txt
│       │   snli_1.0_dev.jsonl
│       │   snli_1.0_dev.txt
│       │   snli_1.0_test.jsonl
│       │   snli_1.0_test.txt
│       │   snli_1.0_test_hard.jsonl
│       │   snli_1.0_train.jsonl
│       │   snli_1.0_train.txt
│       │
│       └───revised_combined
│               test.tsv
├───embeddings
│   └───glove
│           glove.840B.300d.txt
├───processed_data
│   ├───AAN_lra
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_normal.jsonl
│   │       train.jsonl
│   │
│   ├───IMDB
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_contrast.jsonl
│   │       test_counterfactual.jsonl
│   │       test_normal.jsonl
│   │       test_original_of_contrast.jsonl
│   │       test_original_of_counterfactual.jsonl
│   │       train.jsonl
│   │
│   ├───IMDB_lra
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_normal.jsonl
│   │       train.jsonl
│   │
│   ├───listops200speed
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_normal.jsonl
│   │       train.jsonl
│   │
│   ├───listops500speed
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_normal.jsonl
│   │       train.jsonl
│   │
│   ├───listops900speed
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_normal.jsonl
│   │       train.jsonl
│   │
│   ├───listopsc
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_200_300.jsonl
│   │       test_300_400.jsonl
│   │       test_400_500.jsonl
│   │       test_500_600.jsonl
│   │       test_600_700.jsonl
│   │       test_700_800.jsonl
│   │       test_800_900.jsonl
│   │       test_900_1000.jsonl
│   │       test_iid_arg.jsonl
│   │       test_LRA.jsonl
│   │       test_normal.jsonl
│   │       test_ood_10arg.jsonl
│   │       test_ood_15arg.jsonl
│   │       train.jsonl
│   │
│   ├───listopsmix
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_200_300.jsonl
│   │       test_300_400.jsonl
│   │       test_400_500.jsonl
│   │       test_500_600.jsonl
│   │       test_600_700.jsonl
│   │       test_700_800.jsonl
│   │       test_800_900.jsonl
│   │       test_900_1000.jsonl
│   │       test_iid_arg.jsonl
│   │       test_LRA.jsonl
│   │       test_normal.jsonl
│   │       test_ood_10arg.jsonl
│   │       test_ood_15arg.jsonl
│   │       train.jsonl
│   │
│   ├───listops_lra
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_normal.jsonl
│   │       train.jsonl
│   │
│   ├───listops_lra_speed
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_normal.jsonl
│   │       train.jsonl
│   │
│   ├───MNLIdev
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_antonym_matched.jsonl
│   │       test_antonym_mismatched.jsonl
│   │       test_conj_nli.jsonl
│   │       test_content_word_swap_matched.jsonl
│   │       test_content_word_swap_mismatched.jsonl
│   │       test_function_word_swap_matched.jsonl
│   │       test_function_word_swap_mismatched.jsonl
│   │       test_keyboard_swap_matched.jsonl
│   │       test_keyboard_swap_mismatched.jsonl
│   │       test_length_matched.jsonl
│   │       test_length_mismatched.jsonl
│   │       test_matched.jsonl
│   │       test_mismatched.jsonl
│   │       test_negation_matched.jsonl
│   │       test_negation_mismatched.jsonl
│   │       test_numerical.jsonl
│   │       test_swap_matched.jsonl
│   │       test_swap_mismatched.jsonl
│   │       test_word_overlap_matched.jsonl
│   │       test_word_overlap_mismatched.jsonl
│   │       train.jsonl
│   │
│   ├───proplogic
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_0.jsonl
│   │       test_1.jsonl
│   │       test_10.jsonl
│   │       test_11.jsonl
│   │       test_12.jsonl
│   │       test_2.jsonl
│   │       test_3.jsonl
│   │       test_4.jsonl
│   │       test_5.jsonl
│   │       test_6.jsonl
│   │       test_7.jsonl
│   │       test_8.jsonl
│   │       test_9.jsonl
│   │       train.jsonl
│   │
│   ├───QQP
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_normal.jsonl
│   │       test_PAWS_QQP.jsonl
│   │       test_PAWS_WIKI.jsonl
│   │       train.jsonl
│   │
│   ├───SNLI
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       SNLI_dev_normal.jsonl
│   │       SNLI_metadata.pkl
│   │       SNLI_test_hard.jsonl
│   │       SNLI_test_normal.jsonl
│   │       SNLI_train.jsonl
│   │       test_break.jsonl
│   │       test_counterfactual.jsonl
│   │       test_hard.jsonl
│   │       test_normal.jsonl
│   │       train.jsonl
│   │
│   ├───SST2
│   │       dev_normal.jsonl
│   │       metadata.pkl
│   │       test_normal.jsonl
│   │       train.jsonl
│   │
│   └───SST5
│           dev_normal.jsonl
│           metadata.pkl
│           test_normal.jsonl
│           train.jsonl
```
### How to train
Train:
```python trian.py --model=[insert model name] -- dataset=[insert dataset name] --times=[insert total runs] --device=[insert device name] --model_type=[classifier/sentence_pair/sentence_pair2]```

* Check argparser.py for exact options. 
* sentence_pair2 is used for sequence interaction models for sequence matching tasks (NLI, paraphrase detection), otherwise sequence_pair is used for model_type (if nothing about sequence interaction is explicitly mentioned in the paper then we are talking about a different paper).
* Generally we use total times as 3. For LRA we use 2.

### Model Nomenclature
The nomenclature in the codebase and in the paper are a bit different. We provide a mapping here of the form ([codebase model name] == [paper model name])


* CRvNN == CRvNN
* CRvNN_nohalt == CRvNN (during stress test)  
* OM == OM
* GT_GRC == GT-GRC
* EGT_GRC == EGT-GRC  
* BT_GRC == BT-GRC
* BT_GRC_OS == BT-GRC OS (also BT-GRC OneSoft)
* EBT_GRC == EBT-GRC
* EBT_GRC_noslice == EBT-GRC (-slice)  
* EBT_GRC512 == EBT-GRC (512)
* EBT_GRC512 == EBT-GRC (-slice,512)
* GAU_IN == GAU
* EGT_GAU_IN == EBT-GAU
* EBT_GAU_IN == EBT-GAU
* S4DStack == S4D  
* BalancedTreeGRC == BBT-GRC
* HGRC == RIR-GRC
* HCRvNN == RIR-CRvNN
* HOM == RIR-OM
* HEBT_GRC == RIR-EBT-GRC
* HEBT_GRC_noSSM == RIR-EBT-GRC ($-$S4D)
* HEBT_GRC_noRBA == RIR-EBT-GRC ($-$Beam Align)
* HEBT_GRC_random == RIR-EBT-GRC ($+$Random Align)
* HEBT_GRC_small == RIR-EBT-GRC (beam 5)
* HEBT_GRC_chunk20 == RIR-EBT-GRC (chunk 20)
* HEBT_GRC_chunk10 == RIR-EBT-GRC (chunk 10)
* MEGA == MEGA




