# CB4Rec

contextual bandits for user interest exploration

- overleaf doc: https://www.overleaf.com/8491477127bwdmpgvnmxyd 
- data and simulator: [MIND](https://msnews.github.io/); and also [here](https://microsoftapc-my.sharepoint.com/:f:/g/personal/v-mezhang_microsoft_com/EjPH1GwpA4NNn4xv2qhvgy8B-UveTuXVVbKH_WCdyt4A2g?e=YSHG2C)
- Refs:
  - Simulator: [PLM]((https://dl.acm.org/doi/abs/10.1145/3404835.3463069) and corresponding [repo](https://github.com/wuch15/PLM4NewsRec)
  - High related work: hierarchical UCB: https://arxiv.org/pdf/2110.09905.pdf 
  - CB Predictor: [NRMS](https://aclanthology.org/D19-1671/)


## API Ideas

We have offline data and we want to build a CB4Rec out of the offline data such that it works in a live rec as well. We also use the offline data to simulate a live rec system. Thus, the internal representation of the learner should be built on the training data while the simulator is built on the entire offline data independent of the internal representation of the learner. 

* stream_user_encoder, stream_news_encoder: get updated after iteration t (or slower)
* simulator (simulated environment): takes as input user_id and a list of items, outputs a simulated reward: {0,1}  

  * a pre-trained model with a global user encoder and news encoder 

* learner: 

  * sample topic 
  * sample items (sequentially or batched, the top most likely items to a user) @TODO: How to retrieve K most relevant items out of 1 million in practice, without computing the score for each of the 1 mil items 
  * observe (simulated) rewards 
  * update its internal model (topic model, streaming user encoder, streaming news encoder)  



  How item and user representation in online CB should look like? 


## API 

* `core`: Define abstract API (classes)  
  * `contextual_bandit.py`: Abstract CB class and interaction model  
    * `ContextualBanditLearner`: abstract class for CB 
    * `run_contextual_bandit`: interaction loop 
  * `simulator.py`: Abstract class for simulator 
* `algorithms`: Define actual implementations of neural network models, CB algorithms and simulators.  
  * `SingleStageNeuralUCB`
  * `TwoStageNeuralUCB`
  * ... 
  * `nrms_model`: NRMS model 
  * `nrms_sim`: NRMS simulator
* `utils`: Define all utility functions (e.g. data loading, plotting, ...)  
* `configs`: Contains experiment hyperparameter configurations. 
* `unit_test.py`: All unit tests are written (and run) here. 
* `run_experiment.py`: Run experiments here   

## Progress 
* Basic API for all modules that are interacting and working 
* @TODO: 
  * Implement `update` for CBlearner and simulator 
  * Add complicated CBlearners and simulators to the API

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