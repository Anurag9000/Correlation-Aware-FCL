# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 74.47% on 45 observed labels.
Validation accuracy: 0.7502222222222222; seen labels never predicted: 0.
Server-accepted classes: 45; pending private classes: 0.
Accepted-only TEST accuracy: 0.7446666666666667.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **81.0%** (0: 74%, 1: 82%, 2: 87%) | **80.7%** (15: 86%, 16: 67%, 17: 89%) | **79.3%** (30: 80%, 31: 74%, 32: 84%) |
| Client 1 | **69.0%** (3: 59%, 4: 75%, 5: 73%) | **73.7%** (18: 80%, 19: 67%, 20: 74%) | **77.7%** (33: 92%, 34: 74%, 35: 67%) |
| Client 2 | **75.0%** (6: 68%, 7: 82%, 8: 75%) | **81.3%** (21: 99%, 22: 56%, 23: 89%) | **79.7%** (36: 77%, 37: 94%, 38: 68%) |
| Client 3 | **68.0%** (9: 63%, 10: 73%, 11: 68%) | **77.7%** (24: 87%, 25: 74%, 26: 72%) | **66.7%** (39: 62%, 40: 84%, 41: 54%) |
| Client 4 | **73.0%** (12: 74%, 13: 58%, 14: 87%) | **69.0%** (27: 95%, 28: 52%, 29: 60%) | **65.3%** (42: 55%, 43: 67%, 44: 74%) |
