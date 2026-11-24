# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.57% on 14 observed labels.
Validation accuracy: 0.6142857142857143; seen labels never predicted: 1.
Server-accepted classes: 14; pending private classes: 0.
Accepted-only TEST accuracy: 0.6057142857142858.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **73.0%** (0: 47%, 1: 99%) | **50.0%** (10: 100%, 11: 0%) |
| Client 1 | **64.0%** (2: 28%, 3: 100%) | **68.5%** (4: 90%, 5: 47%) |
| Client 2 | **68.5%** (4: 90%, 5: 47%) | **53.5%** (8: 99%, 9: 8%) |
| Client 3 | **61.5%** (6: 27%, 7: 96%) | **53.5%** (12: 92%, 13: 15%) |
| Client 4 | **53.5%** (8: 99%, 9: 8%) | **64.0%** (2: 28%, 3: 100%) |
