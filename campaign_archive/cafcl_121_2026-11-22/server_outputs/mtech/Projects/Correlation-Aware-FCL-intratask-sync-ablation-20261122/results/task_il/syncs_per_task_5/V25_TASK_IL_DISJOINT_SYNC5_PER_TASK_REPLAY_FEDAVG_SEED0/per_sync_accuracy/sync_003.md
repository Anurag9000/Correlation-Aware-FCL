# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 77.20% on 10 observed labels.
Validation accuracy: 0.76; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.772.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **89.0%** (0: 93%, 1: 85%) | **89.0%** (0: 93%, 1: 85%) | **89.0%** (0: 93%, 1: 85%) |
| Client 1 | **87.5%** (2: 78%, 3: 97%) | **87.5%** (2: 78%, 3: 97%) | **87.5%** (2: 78%, 3: 97%) |
| Client 2 | **77.5%** (4: 70%, 5: 85%) | **77.5%** (4: 70%, 5: 85%) | **77.5%** (4: 70%, 5: 85%) |
| Client 3 | **68.0%** (6: 63%, 7: 73%) | **68.0%** (6: 63%, 7: 73%) | **68.0%** (6: 63%, 7: 73%) |
| Client 4 | **64.0%** (8: 97%, 9: 31%) | **64.0%** (8: 97%, 9: 31%) | **64.0%** (8: 97%, 9: 31%) |
