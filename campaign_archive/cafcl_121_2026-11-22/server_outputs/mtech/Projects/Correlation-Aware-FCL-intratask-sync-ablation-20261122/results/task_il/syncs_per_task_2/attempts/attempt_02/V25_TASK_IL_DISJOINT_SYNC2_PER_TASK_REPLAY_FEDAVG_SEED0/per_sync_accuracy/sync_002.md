# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.40% on 10 observed labels.
Validation accuracy: 0.584; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.604.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **76.0%** (0: 52%, 1: 100%) | **76.0%** (0: 52%, 1: 100%) |
| Client 1 | **59.0%** (2: 18%, 3: 100%) | **59.0%** (2: 18%, 3: 100%) |
| Client 2 | **60.5%** (4: 99%, 5: 22%) | **60.5%** (4: 99%, 5: 22%) |
| Client 3 | **53.5%** (6: 8%, 7: 99%) | **53.5%** (6: 8%, 7: 99%) |
| Client 4 | **53.0%** (8: 99%, 9: 7%) | **53.0%** (8: 99%, 9: 7%) |
