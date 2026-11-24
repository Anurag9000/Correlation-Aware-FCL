# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 68.00% on 30 observed labels.
Validation accuracy: 0.6806666666666666; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.7575.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **70.0%** (0: 46%, 1: 94%) | **53.5%** (10: 18%, 11: 89%) [PENDING upload] | **58.5%** (20: 99%, 21: 18%) [PENDING upload] |
| Client 1 | **84.0%** (2: 72%, 3: 96%) | **71.5%** (12: 77%, 13: 66%) | **84.5%** (22: 80%, 23: 89%) |
| Client 2 | **55.0%** (4: 46%, 5: 64%) [PENDING upload] | **47.5%** (14: 92%, 15: 3%) [PENDING upload] | **48.0%** (24: 82%, 25: 14%) [PENDING upload] |
| Client 3 | **72.5%** (6: 49%, 7: 96%) | **82.5%** (16: 77%, 17: 88%) | **78.5%** (26: 60%, 27: 97%) |
| Client 4 | **73.5%** (8: 94%, 9: 53%) | **78.0%** (18: 77%, 19: 79%) | **62.5%** (28: 70%, 29: 55%) |
