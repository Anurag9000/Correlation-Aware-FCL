# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.70% on 10 observed labels.
Validation accuracy: 0.842; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.817.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **86.0%** (0: 92%, 1: 80%) | **86.0%** (0: 92%, 1: 80%) | **86.0%** (0: 92%, 1: 80%) | **86.0%** (0: 92%, 1: 80%) | **86.0%** (0: 92%, 1: 80%) | **86.0%** (0: 92%, 1: 80%) |
| Client 1 | **87.0%** (2: 91%, 3: 83%) | **87.0%** (2: 91%, 3: 83%) | **87.0%** (2: 91%, 3: 83%) | **87.0%** (2: 91%, 3: 83%) | **87.0%** (2: 91%, 3: 83%) | **87.0%** (2: 91%, 3: 83%) |
| Client 2 | **85.0%** (4: 96%, 5: 74%) | **85.0%** (4: 96%, 5: 74%) | **85.0%** (4: 96%, 5: 74%) | **85.0%** (4: 96%, 5: 74%) | **85.0%** (4: 96%, 5: 74%) | **85.0%** (4: 96%, 5: 74%) |
| Client 3 | **71.5%** (6: 59%, 7: 84%) | **71.5%** (6: 59%, 7: 84%) | **71.5%** (6: 59%, 7: 84%) | **71.5%** (6: 59%, 7: 84%) | **71.5%** (6: 59%, 7: 84%) | **71.5%** (6: 59%, 7: 84%) |
| Client 4 | **79.0%** (8: 87%, 9: 71%) | **79.0%** (8: 87%, 9: 71%) | **79.0%** (8: 87%, 9: 71%) | **79.0%** (8: 87%, 9: 71%) | **79.0%** (8: 87%, 9: 71%) | **79.0%** (8: 87%, 9: 71%) |
