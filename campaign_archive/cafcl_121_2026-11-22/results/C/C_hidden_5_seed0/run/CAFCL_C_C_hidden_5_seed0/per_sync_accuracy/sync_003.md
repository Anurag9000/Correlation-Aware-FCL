# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 73.63% on 30 observed labels.
Validation accuracy: 0.7393333333333333; seen labels never predicted: 2.
Server-accepted classes: 24; pending private classes: 6.
Accepted-only TEST accuracy: 0.79125.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **78.0%** (8: 99%, 9: 57%) | **75.0%** (14: 63%, 15: 87%) | **67.0%** (26: 36%, 27: 98%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **50.0%** (18: 0%, 19: 100%) [PENDING upload] | **55.0%** (28: 47%, 29: 63%) [PENDING upload] |
| Client 2 | **83.5%** (4: 81%, 5: 86%) | **93.5%** (16: 88%, 17: 99%) | **81.5%** (22: 71%, 23: 92%) |
| Client 3 | **84.0%** (0: 84%, 1: 84%) | **66.0%** (10: 95%, 11: 37%) | **92.0%** (20: 86%, 21: 98%) |
| Client 4 | **83.5%** (2: 74%, 3: 93%) | **71.5%** (12: 81%, 13: 62%) | **74.0%** (24: 48%, 25: 100%) |
