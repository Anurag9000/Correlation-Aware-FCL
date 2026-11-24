# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 43.92% on 24 observed labels.
Validation accuracy: 0.4275; seen labels never predicted: 0.
Server-accepted classes: 24; pending private classes: 0.
Accepted-only TEST accuracy: 0.43916666666666665.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **62.5%** (0: 84%, 1: 41%) | **52.0%** (4: 46%, 5: 58%) | **41.5%** (8: 38%, 9: 45%) | **58.0%** (12: 62%, 13: 54%) | **65.0%** (16: 54%, 17: 76%) | **19.5%** (20: 20%, 21: 19%) |
| Client 1 | **38.0%** (2: 48%, 3: 28%) | **50.5%** (6: 52%, 7: 49%) | **36.5%** (10: 28%, 11: 45%) | **41.5%** (14: 35%, 15: 48%) | **38.0%** (18: 45%, 19: 31%) | **24.0%** (22: 7%, 23: 41%) |
