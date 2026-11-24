# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.70% on 27 observed labels.
Validation accuracy: 0.6148148148148148; seen labels never predicted: 0.
Server-accepted classes: 27; pending private classes: 0.
Accepted-only TEST accuracy: 0.607037037037037.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **55.6%** (4: 18%, 7: 70%, 15: 55%, 58: 51%, 63: 84%) | **58.0%** (12: 44%, 18: 28%, 71: 89%, 86: 33%, 96: 96%) | **41.6%** (10: 6%, 19: 73%, 45: 34%, 72: 37%, 90: 58%) |
| Client 1 | **50.6%** (4: 18%, 7: 70%, 40: 30%, 58: 51%, 95: 84%) | **50.4%** (12: 44%, 16: 51%, 18: 28%, 86: 33%, 96: 96%) | **43.4%** (10: 6%, 45: 34%, 61: 82%, 72: 37%, 90: 58%) |
| Client 2 | **44.8%** (4: 18%, 7: 70%, 15: 55%, 40: 30%, 58: 51%) | **62.0%** (12: 44%, 55: 87%, 71: 89%, 86: 33%, 91: 57%) | **51.0%** (10: 6%, 19: 73%, 72: 37%, 79: 81%, 90: 58%) |
| Client 3 | **51.4%** (4: 18%, 40: 30%, 58: 51%, 69: 74%, 95: 84%) | **48.4%** (12: 44%, 16: 51%, 18: 28%, 26: 86%, 86: 33%) | **62.8%** (53: 70%, 72: 37%, 79: 81%, 82: 68%, 90: 58%) |
| Client 4 | **59.8%** (4: 18%, 40: 30%, 69: 74%, 80: 93%, 95: 84%) | **50.2%** (12: 44%, 18: 28%, 71: 89%, 86: 33%, 91: 57%) | **50.0%** (10: 6%, 72: 37%, 79: 81%, 82: 68%, 90: 58%) |
