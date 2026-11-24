# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 72.70% on 20 observed labels.
Validation accuracy: 0.722; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.727.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **89.5%** (0: 92%, 1: 87%) | **89.5%** (0: 92%, 1: 87%) | **89.5%** (0: 92%, 1: 87%) | **89.5%** (0: 92%, 1: 87%) | **89.5%** (0: 92%, 1: 87%) | **64.0%** (10: 55%, 11: 73%) |
| Client 1 | **88.5%** (2: 81%, 3: 96%) | **88.5%** (2: 81%, 3: 96%) | **88.5%** (2: 81%, 3: 96%) | **88.5%** (2: 81%, 3: 96%) | **88.5%** (2: 81%, 3: 96%) | **56.5%** (12: 98%, 13: 15%) |
| Client 2 | **84.5%** (4: 93%, 5: 76%) | **84.5%** (4: 93%, 5: 76%) | **84.5%** (4: 93%, 5: 76%) | **84.5%** (4: 93%, 5: 76%) | **84.5%** (4: 93%, 5: 76%) | **52.0%** (14: 99%, 15: 5%) |
| Client 3 | **76.0%** (6: 71%, 7: 81%) | **76.0%** (6: 71%, 7: 81%) | **76.0%** (6: 71%, 7: 81%) | **76.0%** (6: 71%, 7: 81%) | **76.0%** (6: 71%, 7: 81%) | **79.0%** (16: 91%, 17: 67%) |
| Client 4 | **78.5%** (8: 71%, 9: 86%) | **78.5%** (8: 71%, 9: 86%) | **78.5%** (8: 71%, 9: 86%) | **78.5%** (8: 71%, 9: 86%) | **78.5%** (8: 71%, 9: 86%) | **58.5%** (18: 88%, 19: 29%) |
