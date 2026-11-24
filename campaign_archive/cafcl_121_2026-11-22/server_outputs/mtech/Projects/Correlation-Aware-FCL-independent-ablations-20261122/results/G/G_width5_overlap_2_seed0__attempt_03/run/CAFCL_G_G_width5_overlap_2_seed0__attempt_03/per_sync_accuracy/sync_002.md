# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 47.65% on 34 observed labels.
Validation accuracy: 0.4747058823529412; seen labels never predicted: 1.
Server-accepted classes: 34; pending private classes: 0.
Accepted-only TEST accuracy: 0.4764705882352941.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **42.6%** (4: 26%, 7: 55%, 15: 25%, 58: 17%, 63: 90%) | **39.4%** (1: 0%, 21: 13%, 64: 71%, 93: 82%, 94: 31%) |
| Client 1 | **56.0%** (16: 8%, 69: 53%, 80: 97%, 86: 36%, 95: 86%) | **36.6%** (10: 1%, 19: 38%, 45: 0%, 72: 48%, 79: 96%) |
| Client 2 | **41.4%** (4: 26%, 15: 25%, 40: 17%, 69: 53%, 95: 86%) | **31.6%** (10: 1%, 45: 0%, 55: 94%, 61: 47%, 90: 16%) |
| Client 3 | **54.8%** (12: 65%, 16: 8%, 18: 89%, 26: 59%, 69: 53%) | **24.6%** (1: 0%, 45: 0%, 64: 71%, 82: 11%, 92: 41%) |
| Client 4 | **56.0%** (16: 8%, 69: 53%, 71: 67%, 91: 75%, 96: 77%) | **29.6%** (44: 80%, 45: 0%, 53: 9%, 72: 48%, 82: 11%) |
