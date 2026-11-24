# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 62.03% on 30 observed labels.
Validation accuracy: 0.6106666666666667; seen labels never predicted: 0.
Server-accepted classes: 30; pending private classes: 0.
Accepted-only TEST accuracy: 0.6203333333333333.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **69.5%** (0: 41%, 1: 98%) | **52.5%** (10: 90%, 11: 15%) | **79.0%** (20: 58%, 21: 100%) |
| Client 1 | **70.0%** (2: 40%, 3: 100%) | **52.5%** (12: 95%, 13: 10%) | **77.0%** (22: 74%, 23: 80%) |
| Client 2 | **70.5%** (4: 93%, 5: 48%) | **59.0%** (14: 30%, 15: 88%) | **70.0%** (24: 82%, 25: 58%) |
| Client 3 | **52.5%** (6: 8%, 7: 97%) | **61.0%** (16: 22%, 17: 100%) | **59.5%** (26: 20%, 27: 99%) |
| Client 4 | **54.5%** (8: 97%, 9: 12%) | **50.0%** (18: 96%, 19: 4%) | **53.0%** (28: 99%, 29: 7%) |
