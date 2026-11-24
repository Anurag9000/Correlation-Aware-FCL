# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 71.20% on 10 observed labels.
Validation accuracy: 0.706; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.712.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **83.5%** (0: 80%, 1: 87%) | **83.5%** (0: 80%, 1: 87%) | **83.5%** (0: 80%, 1: 87%) | **83.5%** (0: 80%, 1: 87%) |
| Client 1 | **78.5%** (2: 58%, 3: 99%) | **78.5%** (2: 58%, 3: 99%) | **78.5%** (2: 58%, 3: 99%) | **78.5%** (2: 58%, 3: 99%) |
| Client 2 | **70.5%** (4: 93%, 5: 48%) | **70.5%** (4: 93%, 5: 48%) | **70.5%** (4: 93%, 5: 48%) | **70.5%** (4: 93%, 5: 48%) |
| Client 3 | **66.5%** (6: 70%, 7: 63%) | **66.5%** (6: 70%, 7: 63%) | **66.5%** (6: 70%, 7: 63%) | **66.5%** (6: 70%, 7: 63%) |
| Client 4 | **57.0%** (8: 97%, 9: 17%) | **57.0%** (8: 97%, 9: 17%) | **57.0%** (8: 97%, 9: 17%) | **57.0%** (8: 97%, 9: 17%) |
