# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 51.96% on 50 observed labels.
Validation accuracy: 0.5076; seen labels never predicted: 1.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.5196.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **52.0%** (0: 25%, 1: 29%, 2: 66%, 3: 66%, 4: 74%) | **31.2%** (25: 11%, 26: 4%, 27: 99%, 28: 29%, 29: 13%) |
| Client 1 | **59.8%** (5: 70%, 6: 59%, 7: 47%, 8: 93%, 9: 30%) | **49.2%** (30: 62%, 31: 92%, 32: 4%, 33: 22%, 34: 66%) |
| Client 2 | **57.4%** (10: 51%, 11: 58%, 12: 88%, 13: 47%, 14: 43%) | **51.6%** (35: 22%, 36: 22%, 37: 76%, 38: 85%, 39: 53%) |
| Client 3 | **53.8%** (15: 52%, 16: 31%, 17: 78%, 18: 97%, 19: 11%) | **39.0%** (40: 22%, 41: 0%, 42: 61%, 43: 32%, 44: 80%) |
| Client 4 | **67.0%** (20: 20%, 21: 94%, 22: 70%, 23: 81%, 24: 70%) | **58.6%** (45: 83%, 46: 53%, 47: 48%, 48: 31%, 49: 78%) |
