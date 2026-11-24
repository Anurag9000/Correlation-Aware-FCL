# Server sync 8 — global model v8

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 87.59% on 32 observed labels.
Validation accuracy: 0.8625; seen labels never predicted: 0.
Server-accepted classes: 32; pending private classes: 0.
Accepted-only TEST accuracy: 0.8759375.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **88.5%** (0: 93%, 1: 84%) | **84.5%** (4: 96%, 5: 73%) | **87.0%** (8: 93%, 9: 81%) | **83.0%** (12: 90%, 13: 76%) | **95.0%** (16: 93%, 17: 97%) | **94.5%** (20: 92%, 21: 97%) | **88.5%** (24: 94%, 25: 83%) | **90.0%** (28: 86%, 29: 94%) |
| Client 1 | **88.5%** (2: 83%, 3: 94%) | **78.0%** (6: 70%, 7: 86%) | **82.0%** (10: 78%, 11: 86%) | **87.0%** (14: 87%, 15: 87%) | **84.0%** (18: 92%, 19: 76%) | **92.5%** (22: 88%, 23: 97%) | **82.0%** (26: 82%, 27: 82%) | **96.5%** (30: 100%, 31: 93%) |
