# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 52.90% on 30 observed labels.
Validation accuracy: 0.5293333333333333; seen labels never predicted: 3.
Server-accepted classes: 20; pending private classes: 10.
Accepted-only TEST accuracy: 0.512.
Terminal flush: False. Selected clients: [1, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **68.0%** (0: 60%, 1: 76%) [PENDING upload] | **51.5%** (10: 53%, 11: 50%) [PENDING upload] | **50.0%** (20: 100%, 21: 0%) [PENDING upload] |
| Client 1 | **81.5%** (2: 71%, 3: 92%) | **50.0%** (12: 100%, 13: 0%) | **40.0%** (22: 59%, 23: 21%) |
| Client 2 | **50.0%** (4: 100%, 5: 0%) | **55.0%** (14: 57%, 15: 53%) | **58.5%** (24: 88%, 25: 29%) [PENDING upload] |
| Client 3 | **50.0%** (6: 23%, 7: 77%) | **34.5%** (16: 56%, 17: 13%) | **43.5%** (26: 84%, 27: 3%) |
| Client 4 | **65.0%** (8: 89%, 9: 41%) | **42.5%** (18: 19%, 19: 66%) | **53.5%** (28: 63%, 29: 44%) [PENDING upload] |
