# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.85% on 20 observed labels.
Validation accuracy: 0.606; seen labels never predicted: 1.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.6185.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **73.0%** (0: 53%, 1: 93%) | **51.0%** (10: 87%, 11: 15%) |
| Client 1 | **64.5%** (2: 29%, 3: 100%) | **50.0%** (12: 0%, 13: 100%) |
| Client 2 | **75.5%** (4: 85%, 5: 66%) | **65.0%** (14: 61%, 15: 69%) |
| Client 3 | **56.0%** (6: 18%, 7: 94%) | **77.5%** (16: 61%, 17: 94%) |
| Client 4 | **52.5%** (8: 99%, 9: 6%) | **53.5%** (18: 31%, 19: 76%) |
