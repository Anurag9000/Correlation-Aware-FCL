# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 69.27% on 11 observed labels.
Validation accuracy: 0.6581818181818182; seen labels never predicted: 0.
Server-accepted classes: 11; pending private classes: 0.
Accepted-only TEST accuracy: 0.6927272727272727.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **79.5%** (0: 75%, 1: 84%) | **57.5%** (5: 37%, 6: 78%) |
| Client 1 | **63.5%** (1: 84%, 2: 43%) | **74.5%** (6: 78%, 7: 71%) |
| Client 2 | **67.5%** (2: 43%, 3: 92%) | **84.5%** (7: 71%, 8: 98%) |
| Client 3 | **87.5%** (3: 92%, 4: 83%) | **49.5%** (8: 98%, 9: 1%) |
| Client 4 | **60.0%** (4: 83%, 5: 37%) | **50.5%** (9: 1%, 10: 100%) |
