# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.40% on 40 observed labels.
Validation accuracy: 0.768; seen labels never predicted: 1.
Server-accepted classes: 36; pending private classes: 4.
Accepted-only TEST accuracy: 0.8066666666666666.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **83.5%** (8: 83%, 9: 84%) | **73.5%** (14: 78%, 15: 69%) | **77.0%** (26: 66%, 27: 88%) | **77.5%** (34: 90%, 35: 65%) |
| Client 1 | **72.0%** (6: 64%, 7: 80%) | **77.0%** (18: 60%, 19: 94%) | **76.5%** (28: 79%, 29: 74%) | **93.5%** (30: 98%, 31: 89%) |
| Client 2 | **82.0%** (4: 93%, 5: 71%) | **91.0%** (16: 82%, 17: 100%) | **84.0%** (22: 71%, 23: 97%) | **76.0%** (32: 58%, 33: 94%) |
| Client 3 | **88.0%** (0: 82%, 1: 94%) | **76.5%** (10: 85%, 11: 68%) | **93.0%** (20: 86%, 21: 100%) | **80.5%** (38: 92%, 39: 69%) |
| Client 4 | **83.0%** (2: 69%, 3: 97%) | **67.5%** (12: 86%, 13: 49%) | **50.0%** (24: 100%, 25: 0%) [PENDING upload] | **46.0%** (36: 91%, 37: 1%) [PENDING upload] |
