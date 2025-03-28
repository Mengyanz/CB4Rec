# Supplementary Code for Two-Stage Neural Contextual Bandits for Adaptive Personalised Recommendation.

## API 

* `core`: Define abstract API (classes)  
  * `contextual_bandit.py`: Abstract CB class and interaction model  
    * `ContextualBanditLearner`: Abstract class for CB 
    * `run_contextual_bandit`: Interaction loop 
  * `simulator.py`: Abstract class for simulator 
* `algorithms`: Define actual implementations of neural network models, CB algorithms and simulators.  
  * `linucb`: Linucb and glmucb
  * `neural_linear`: Neural linucb and glmucb, additive glmucb
  * `neural_bilinear`: Neural generalised bilinear ucb
  * `neural_greedy`: One and two stage neural greedy 
  * `neural_ucb`: One and two stage neural dropout ucb
  * `proposed`: Two stage generalised additive and bilinear ucb
  * `uniform_ucb`: One and two stage random
  * `nrms_model`: NRMS model 
  * `nrms_sim`: NRMS simulator
  * `lr_model`: Logistic regression model
* `utils`: Define all utility functions (e.g. data loading, plotting, ...)  
* `configs`: Contains experiment hyperparameter configurations. 
* `run_experiment`: Run single experiment
* `tune_experiment`: Run group of experiments in parallell on gpus
* `preprocess.py`: word2vec using glove6B and split data for training simulators and CB learners. Simulators are trained on the train data and are selected on the valid data. For CB simulation, we do as follows: 
    * In each trial out of `args.n_trials` (i.e. the number of experiments), we randomly select `args.num_selected_users` from the train set. The first portion of the train set with all the behaviour data of those `args.num_selected_users` removed is used to pre-train the internal model of a CB learner. The first portion of the train set is controlled by `args.cb_train_ratio`
    * Then, at each iteration, we randomly sample a user from the set of `args.num_selected_users` users and obtain the user's context from the other portion of the train set at the iteration. A CB learner observes the user's context and produces recommendations. 
    * Note that in each trial, we randomly generate a different set of `args.num_selected_users` users. Thus, we need to retrain the internal model of the CB learner in each trial. 



## Usage 

- Step 1: download data [MIND-Large](https://msnews.github.io/)/[movielen-20M](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset) and word embeddings ([GloVe](https://nlp.stanford.edu/projects/glove/) in `./data/`. Convert MovieLens-20M using `movielens2mind.ipynb`. Data folder structure as:
```bash
data/dataset_name
├── train
│   ├── news.csv             
│   ├── behaviors.csv
├── valid                   
│   ├── news.csv             
│   ├── behaviors.csv
├── utils                   
│   ├── ....
```
- Step 2: download [utils](https://release.after.accept) in `/data/large/utils` and [pretrained models](https://release.after.accept) in `./CB4Rec`. Alternatively, you can pre-processing data and pretrain models follow `preprocess.py`.
- Step 3: specify parameter settings in `tune_experiment.py` and add the commands you want to run in `create_commands` as needed. 
- Step 4: run `python tune_experiment.py` in command line. Evalautions will be done after the simulations and saved.