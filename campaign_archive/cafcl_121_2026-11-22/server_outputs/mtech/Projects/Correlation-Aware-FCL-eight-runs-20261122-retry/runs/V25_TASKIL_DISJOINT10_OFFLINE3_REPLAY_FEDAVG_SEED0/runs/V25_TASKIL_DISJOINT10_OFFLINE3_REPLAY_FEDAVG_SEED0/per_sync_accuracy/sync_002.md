# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 56.45% on 20 observed labels.
Validation accuracy: 0.579; seen labels never predicted: 2.
Server-accepted classes: 10; pending private classes: 10.
Accepted-only TEST accuracy: 0.594.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **54.0%** (0: 9%, 1: 99%) [PENDING upload] | **48.5%** (10: 40%, 11: 57%) [PENDING upload] |
| Client 1 | **77.5%** (2: 88%, 3: 67%) | **49.5%** (12: 13%, 13: 86%) [PENDING upload] |
| Client 2 | **50.0%** (4: 100%, 5: 0%) | **59.5%** (14: 56%, 15: 63%) |
| Client 3 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **65.5%** (16: 64%, 17: 67%) [PENDING upload] |
| Client 4 | **64.0%** (8: 78%, 9: 50%) | **46.0%** (18: 18%, 19: 74%) |
