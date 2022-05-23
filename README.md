# Supplementary Code for Two-Stage Neural Contextual Bandits for Personalised News Recommendation.

## API 

* `core`: Define abstract API (classes)  
  * `contextual_bandit.py`: Abstract CB class and interaction model  
    * `ContextualBanditLearner`: abstract class for CB 
    * `run_contextual_bandit`: interaction loop 
  * `simulator.py`: Abstract class for simulator 
* `algorithms`: Define actual implementations of neural network models, CB algorithms and simulators.  
  * `NeuralDropoutUCB`
  * ... 
  * `nrms_model`: NRMS model 
  * `nrms_sim`: NRMS simulator
* `utils`: Define all utility functions (e.g. data loading, plotting, ...)  
* `configs`: Contains experiment hyperparameter configurations. 
* `run_experiment.py`: Run experiments here   
* `preprocess.py`: word2vec using glove6B and split data for training simulators and CB learners. Simulators are trained on the MIND train data and are selected on the MIND valid data. For CB simulation, we do as follows: 
    * In each trial out of `args.n_trials` (i.e. the number of experiments), we randomly select `args.num_selected_users` from the MIND train set. The first portion of the MIND train set with all the behaviour data of those `args.num_selected_users` removed is used to pre-train the internal model of a CB learner. The first portion of the MIND train set is controlled by `args.cb_train_ratio`
    * Then, at each iteration, we randomly sample a user from the set of `args.num_selected_users` users and obtain the user's context from the other portion of the MIND train set at the iteration. A CB learner observes the user's context and produces recommendations. 
    * Note that in each trial, we randomly generate a different set of `args.num_selected_users` users. Thus, we need to retrain the internal model of the CB learner in each trial. 

    * With the MIND data statistics as below: 

|             | train       | valid   | intersection | 
| ----------- | ----------- |---------|--------------|
| # of users  | 711,222     |  255,990|216,778       |
| # of samples| 2,232,748   |  376,471|N/A           |



## Usage 

- Step 1: download [data](), [pretrained models and utils]() and specify the root path in `tune_experiment.py`.
- Step 2: specify parameter settings in `tune_experiment.py` and add the commands you want to run in `create_commands` as needed. 
- Step 3: run `python tune_experiment.py` in command line. Evalautions will be done after the simulations and saved.