# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 84.34% on 50 observed labels.
Validation accuracy: 0.8512; seen labels never predicted: 0.
Server-accepted classes: 50; pending private classes: 0.
Accepted-only TEST accuracy: 0.8434.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **87.0%** (0: 92%, 1: 82%) | **68.0%** (10: 93%, 11: 43%) | **94.0%** (20: 93%, 21: 95%) | **96.5%** (30: 96%, 31: 97%) | **84.0%** (40: 94%, 41: 74%) |
| Client 1 | **88.0%** (2: 82%, 3: 94%) | **75.0%** (12: 86%, 13: 64%) | **87.5%** (22: 82%, 23: 93%) | **84.0%** (32: 72%, 33: 96%) | **73.0%** (42: 58%, 43: 88%) |
| Client 2 | **89.0%** (4: 91%, 5: 87%) | **76.5%** (14: 71%, 15: 82%) | **87.0%** (24: 85%, 25: 89%) | **78.5%** (34: 93%, 35: 64%) | **76.5%** (44: 92%, 45: 61%) |
| Client 3 | **82.5%** (6: 82%, 7: 83%) | **90.5%** (16: 87%, 17: 94%) | **78.5%** (26: 71%, 27: 86%) | **88.5%** (36: 89%, 37: 88%) | **94.5%** (46: 98%, 47: 91%) |
| Client 4 | **89.5%** (8: 89%, 9: 90%) | **85.5%** (18: 88%, 19: 83%) | **78.0%** (28: 84%, 29: 72%) | **82.5%** (38: 91%, 39: 74%) | **94.0%** (48: 89%, 49: 99%) |
