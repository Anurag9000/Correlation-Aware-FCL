# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 67.60% on 10 observed labels.
Validation accuracy: 0.652; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.676.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **81.5%** (0: 68%, 1: 95%) | **81.5%** (0: 68%, 1: 95%) |
| Client 1 | **69.0%** (2: 38%, 3: 100%) | **69.0%** (2: 38%, 3: 100%) |
| Client 2 | **62.5%** (4: 99%, 5: 26%) | **62.5%** (4: 99%, 5: 26%) |
| Client 3 | **67.5%** (6: 69%, 7: 66%) | **67.5%** (6: 69%, 7: 66%) |
| Client 4 | **57.5%** (8: 96%, 9: 19%) | **57.5%** (8: 96%, 9: 19%) |
