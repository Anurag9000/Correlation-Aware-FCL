# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 56.82% on 11 observed labels.
Validation accuracy: 0.56; seen labels never predicted: 0.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.5936363636363636.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **54.0%** (0: 89%, 1: 19%) | **74.5%** (5: 53%, 6: 96%) |
| Client 1 | **56.0%** (1: 19%, 2: 93%) | **48.0%** (6: 96%, 7: 0%) |
| Client 2 | **46.5%** (2: 93%, 3: 0%) | **39.0%** (7: 0%, 8: 78%) |
| Client 3 | **44.0%** (3: 0%, 4: 88%) | **47.0%** (8: 78%, 9: 16%) |
| Client 4 | **70.5%** (4: 88%, 5: 53%) | **54.5%** (9: 16%, 10: 93%) |
