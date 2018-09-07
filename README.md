# tempoGAN

Implementation of [tempoGAN: A Temporally Coherent, Volumetric GAN for Super-resolution Fluid
Flow](https://arxiv.org/abs/1801.09710).

1. Data generation:
```
manta ./data/data_generator.py
python3 ./data/torecord.py --output traindata.tfrecords
```

2. Training:
```
python3 ./src/train.py
```

Implemented by
* [Stephan Antholzer](https://github.com/antholzer)
* [Johannes Sappl](https://github.com/johannessappl)
* [Hessel Tuinhof](https://github.com/hesseltuinhof)
