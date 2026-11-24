# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 83.30% on 10 observed labels.
Validation accuracy: 0.808; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.833.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **88.5%** (0: 93%, 1: 84%) | **88.5%** (0: 93%, 1: 84%) | **88.5%** (0: 93%, 1: 84%) | **88.5%** (0: 93%, 1: 84%) | **88.5%** (0: 93%, 1: 84%) |
| Client 1 | **89.5%** (2: 80%, 3: 99%) | **89.5%** (2: 80%, 3: 99%) | **89.5%** (2: 80%, 3: 99%) | **89.5%** (2: 80%, 3: 99%) | **89.5%** (2: 80%, 3: 99%) |
| Client 2 | **80.5%** (4: 80%, 5: 81%) | **80.5%** (4: 80%, 5: 81%) | **80.5%** (4: 80%, 5: 81%) | **80.5%** (4: 80%, 5: 81%) | **80.5%** (4: 80%, 5: 81%) |
| Client 3 | **79.0%** (6: 65%, 7: 93%) | **79.0%** (6: 65%, 7: 93%) | **79.0%** (6: 65%, 7: 93%) | **79.0%** (6: 65%, 7: 93%) | **79.0%** (6: 65%, 7: 93%) |
| Client 4 | **79.0%** (8: 90%, 9: 68%) | **79.0%** (8: 90%, 9: 68%) | **79.0%** (8: 90%, 9: 68%) | **79.0%** (8: 90%, 9: 68%) | **79.0%** (8: 90%, 9: 68%) |
