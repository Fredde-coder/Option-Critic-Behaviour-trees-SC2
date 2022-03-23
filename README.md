# Option Critic with Behaviour trees in Starcraft 2
This repository is the foundation for my [thesis](http://kth.diva-portal.org/smash/record.jsf?aq2=%5B%5B%5D%5D&c=7&af=%5B%5D&searchType=LIST_LATEST&sortOrder2=title_sort_asc&query=&language=en&pid=diva2%3A1640875&aq=%5B%5B%5D%5D&sf=all&aqe=%5B%5D&sortOrder=author_sort_asc&onlyFullText=false&noOfRows=50&dswid=2044) work and is an extension of Laurens Weitkamps PyTorch [implementation](https://github.com/lweitkamp/option-critic-pytorch) of the paper "The Option-Critic Architecture" by Pierre-Luc Bacon, Jean Harb and Doina Precup [arXiv](https://arxiv.org/abs/1609.05140) by implementing a new feature network after Deepminds convolutional archietcture [arXiv](https://arxiv.org/abs/1708.04782) to adapt Option-Critic (OC) for the Starcraft 2 environment. 

The other contribution is the implementation of two OC behaviour trees (BTs) combination models. One with BTs as actions (OCBT) and the other with BTs as options (OCBToptions). Currently these are implemented for the Build marines and Defeat zerglings and banelings mini-games, but can be developed for the full game as well as the other mini-games fairly simply. However, that means that only regular OC is possible in other mini-games or the full game currently. 

## Build marines
Build marines is the default experiment and regular OC the default model:
```python main.py ```

To run the OCBT model:
```python main.py --bt-comb "true"```

To run the OCBToptions model:
```python main.py --partial-bt 2```
## Defeat zerglings and banelings
To run the Defeat zerlings and banelings experiment:
```python main.py --env-kwargs '{\"map_name\":\"DefeatZerglingsAndBanelings\", \"extract\":true,\"step_mul\":16 , \"visualize\":false,\"get_action_spec\":true, \"reduced_input\":false,\"reduce_actions\":false}'```


## Four Room experiment
To run the four room experiment:
```python main.py --env "fourrooms" --env-kwargs '{\"env_seed\":true, \"map_name\":\"fourrooms\"}' --avoid-bad-env-init "false"```
## Arguments
For a full list of arguments with explanations checkout the beginning of main.py

## Requirements
Starcraft 2 installed (it is currently free via Blizzard)

```
pytorch 1.9.0
tensorboard 2.6.0
gym 0.15.3
pysc2 1.2
py-trees 2.1.6
```
