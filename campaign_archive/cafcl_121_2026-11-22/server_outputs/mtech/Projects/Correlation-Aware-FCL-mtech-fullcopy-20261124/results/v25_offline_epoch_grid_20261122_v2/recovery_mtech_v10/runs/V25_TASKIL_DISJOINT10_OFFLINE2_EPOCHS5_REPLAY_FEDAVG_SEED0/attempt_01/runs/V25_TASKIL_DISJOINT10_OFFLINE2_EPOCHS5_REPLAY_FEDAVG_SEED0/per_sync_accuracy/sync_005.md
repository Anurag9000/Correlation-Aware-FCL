# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 71.56% on 50 observed labels.
Validation accuracy: 0.71; seen labels never predicted: 1.
Server-accepted classes: 38; pending private classes: 12.
Accepted-only TEST accuracy: 0.7855263157894737.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **73.0%** (0: 50%, 1: 96%) | **55.5%** (10: 92%, 11: 19%) | **82.0%** (20: 67%, 21: 97%) | **88.5%** (30: 94%, 31: 83%) | **52.0%** (40: 18%, 41: 86%) [PENDING upload] |
| Client 1 | **88.5%** (2: 90%, 3: 87%) | **73.5%** (12: 84%, 13: 63%) | **82.5%** (22: 73%, 23: 92%) | **80.5%** (32: 91%, 33: 70%) | **68.0%** (42: 91%, 43: 45%) |
| Client 2 | **38.5%** (4: 73%, 5: 4%) [PENDING upload] | **47.0%** (14: 65%, 15: 29%) [PENDING upload] | **46.0%** (24: 59%, 25: 33%) [PENDING upload] | **63.0%** (34: 77%, 35: 49%) [PENDING upload] | **50.0%** (44: 0%, 45: 100%) [PENDING upload] |
| Client 3 | **76.5%** (6: 72%, 7: 81%) | **87.5%** (16: 87%, 17: 88%) | **74.5%** (26: 54%, 27: 95%) | **86.0%** (36: 83%, 37: 89%) | **75.0%** (46: 99%, 47: 51%) |
| Client 4 | **80.0%** (8: 91%, 9: 69%) | **78.0%** (18: 93%, 19: 63%) | **80.5%** (28: 90%, 29: 71%) | **69.5%** (38: 87%, 39: 52%) | **93.0%** (48: 88%, 49: 98%) |
