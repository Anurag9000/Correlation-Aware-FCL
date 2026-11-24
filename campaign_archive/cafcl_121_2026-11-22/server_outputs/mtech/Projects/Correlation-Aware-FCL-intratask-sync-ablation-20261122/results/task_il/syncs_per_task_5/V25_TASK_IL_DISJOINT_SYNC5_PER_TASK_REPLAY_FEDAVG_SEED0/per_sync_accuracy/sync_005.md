# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 83.40% on 10 observed labels.
Validation accuracy: 0.808; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.834.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **90.0%** (0: 88%, 1: 92%) | **90.0%** (0: 88%, 1: 92%) | **90.0%** (0: 88%, 1: 92%) | **90.0%** (0: 88%, 1: 92%) | **90.0%** (0: 88%, 1: 92%) |
| Client 1 | **87.5%** (2: 76%, 3: 99%) | **87.5%** (2: 76%, 3: 99%) | **87.5%** (2: 76%, 3: 99%) | **87.5%** (2: 76%, 3: 99%) | **87.5%** (2: 76%, 3: 99%) |
| Client 2 | **83.0%** (4: 91%, 5: 75%) | **83.0%** (4: 91%, 5: 75%) | **83.0%** (4: 91%, 5: 75%) | **83.0%** (4: 91%, 5: 75%) | **83.0%** (4: 91%, 5: 75%) |
| Client 3 | **79.5%** (6: 70%, 7: 89%) | **79.5%** (6: 70%, 7: 89%) | **79.5%** (6: 70%, 7: 89%) | **79.5%** (6: 70%, 7: 89%) | **79.5%** (6: 70%, 7: 89%) |
| Client 4 | **77.0%** (8: 94%, 9: 60%) | **77.0%** (8: 94%, 9: 60%) | **77.0%** (8: 94%, 9: 60%) | **77.0%** (8: 94%, 9: 60%) | **77.0%** (8: 94%, 9: 60%) |
