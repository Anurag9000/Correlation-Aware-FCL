# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.11% on 28 observed labels.
Validation accuracy: 0.8271428571428572; seen labels never predicted: 0.
Server-accepted classes: 28; pending private classes: 0.
Accepted-only TEST accuracy: 0.8110714285714286.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **85.5%** (0: 82%, 1: 89%) | **77.5%** (8: 96%, 9: 59%) | **93.5%** (16: 90%, 17: 97%) | **80.0%** (22: 66%, 23: 94%) |
| Client 1 | **85.0%** (2: 72%, 3: 98%) | **67.5%** (10: 88%, 11: 47%) | **77.5%** (8: 96%, 9: 59%) | **93.5%** (16: 90%, 17: 97%) |
| Client 2 | **86.0%** (4: 83%, 5: 89%) | **70.5%** (12: 89%, 13: 52%) | **82.5%** (18: 81%, 19: 84%) | **90.5%** (20: 81%, 21: 100%) |
| Client 3 | **77.0%** (6: 62%, 7: 92%) | **82.0%** (14: 81%, 15: 83%) | **90.5%** (20: 81%, 21: 100%) | **87.0%** (24: 82%, 25: 92%) |
| Client 4 | **77.5%** (8: 96%, 9: 59%) | **86.0%** (4: 83%, 5: 89%) | **70.5%** (12: 89%, 13: 52%) | **71.0%** (26: 43%, 27: 99%) |
