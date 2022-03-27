# CB4Rec

contextual bandits for user interest exploration

- overleaf doc: https://www.overleaf.com/8491477127bwdmpgvnmxyd 
- data and simulator: [MIND](https://msnews.github.io/); and also [onedrive](https://microsoftapc-my.sharepoint.com/:f:/g/personal/v-mezhang_microsoft_com/EjPH1GwpA4NNn4xv2qhvgy8B-UveTuXVVbKH_WCdyt4A2g?e=YSHG2C)
- Refs:
  - Simulator: [PLM]((https://dl.acm.org/doi/abs/10.1145/3404835.3463069) and corresponding [repo](https://github.com/wuch15/PLM4NewsRec)
  - High related work: hierarchical UCB: https://arxiv.org/pdf/2110.09905.pdf 
  - CB Predictor: [NRMS](https://aclanthology.org/D19-1671/)



## API 

* `core`: Define abstract API (classes)  
  * `contextual_bandit.py`: Abstract CB class and interaction model  
    * `ContextualBanditLearner`: abstract class for CB 
    * `run_contextual_bandit`: interaction loop 
  * `simulator.py`: Abstract class for simulator 
* `algorithms`: Define actual implementations of neural network models, CB algorithms and simulators.  
  * `NeuralDropoutUCB`
  * `TwoStageNeuralUCB`
  * ... 
  * `nrms_model`: NRMS model 
  * `nrms_sim`: NRMS simulator
* `utils`: Define all utility functions (e.g. data loading, plotting, ...)  
* `configs`: Contains experiment hyperparameter configurations. 
* `unit_test.py`: All unit tests are written (and run) here. 
* `run_experiment.py`: Run experiments here   
* `thanh_preprocess.py`: word2vec using glove6B and split data for training simulators and CB learners. Simulators are trained on the MIND train data and are selected on the MIND valid data. For CB simulation, we do as follows: 
    * In each trial out of `args.n_trials` (i.e. the number of experiments), we randomly select `args.num_selected_users` from the MIND train set. The first portion of the MIND train set with all the behaviour data of those `args.num_selected_users` removed is used to pre-train the internal model of a CB learner. The first portion of the MIND train set is controlled by `args.cb_train_ratio`
    * Then, at each iteration, we randomly sample a user from the set of `args.num_selected_users` users and obtain the user's context from the other portion of the MIND train set at the iteration. A CB learner observes the user's context and produces recommendations. 
    * Note that in each trial, we randomly generate a different set of `args.num_selected_users` users. Thus, we need to retrain the internal model of the CB learner in each trial. 

    * With the MIND data statistics as below: 

|             | train       | valid   | intersection | 
| ----------- | ----------- |---------|--------------|
| # of users  | 711,222     |  255,990|216,778       |
| # of samples| 2,232,748   |  376,471|N/A           |

## Guideline for writing classes/functions 
* Try to explain the function and define their args and returns. Eg. 
```
def f(arg1, arg2, arg3, flag):
    """Compute element-wise sum. """

    Args:
      arg1: (None, d) tensor, input 1 
      arg2: (None, d) tensor, input 2
      arg2: (None, d) tensor, input 3
      flag: bool, a flag to trigger something 

    Return:
      out: (None, d) 
```