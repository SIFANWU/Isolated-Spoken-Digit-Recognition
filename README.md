# Isolated-Spoken-Digit-Recognition
This individual assignment is about implementing, training and evaluating Hidden Markov Models (HMMs) for recognition of isolated spoken digits.


Task Description
---------
The speech data is selected from the TI-Digits corpus and consists of isolated digits spoken by a large number of speakers of both genders. There are 11 digits in total: “zero”, “oh”, and “1-9”. The data is split into 2 exclusive sets: a training set and a test set. The training set is to be used for training the model parameters (100 utterance per digit). The test set should be used for testing only (60 utterances per digit). The list of utterances in each data set can be found in

*data/flists/flist_train.txt</br>
*data/flists/flist_test.txt

Items comment
----
hmm.py ：Defines a basic left-to-right no-skip HMM and employs Viterbi to find the most probable path through the HMM given a sequence of
         observations. Function viterbi_decoding is incomplete.</br>
</br>         
hmm_train.py : Performs Viterbi-style HMM training using viterbi_decoding from hmm.py . Function viterbi_train is incomplete.</br>
</br> 
hmm_eval.py : Performs evaluation of HMMs using viterbi_decoding from hmm.py . Function “ eval_HMMs ” is incomplete.</br>
</br> 
gmm_demo.py : Demonstrates the use of Gaussian Mixture Models (GMMs) for this task. This program serves as a baseline for comparison with               the HMM-based recogniser. <br>
</br> 
test_viterbi.py : Uses a set of pre-trained HMMs to test viterbi_decoding from hmm.py . For a test signal it computes the log-likelihood                     score and the most probable state sequence for the corresponding HMM and a mismatched HMM. The correct HMM would produce                   a higher score and a more reasonable state sequence.</br>
</br> 
speechtech/ : A folder containing auxiliary functions needed for this task. You don’t need to modify anything in this folder.
