# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.73% on 22 observed labels.
Validation accuracy: 0.7609090909090909; seen labels never predicted: 0.
Server-accepted classes: 22; pending private classes: 0.
Accepted-only TEST accuracy: 0.7572727272727273.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **81.0%** (0: 76%, 1: 86%) | **63.0%** (8: 98%, 9: 28%) | **77.0%** (16: 54%, 17: 100%) |
| Client 1 | **83.5%** (2: 70%, 3: 97%) | **64.0%** (10: 78%, 11: 50%) | **63.0%** (8: 98%, 9: 28%) |
| Client 2 | **83.5%** (4: 90%, 5: 77%) | **60.0%** (12: 88%, 13: 32%) | **79.5%** (18: 66%, 19: 93%) |
| Client 3 | **76.5%** (6: 60%, 7: 93%) | **74.5%** (14: 50%, 15: 99%) | **90.5%** (20: 89%, 21: 92%) |
| Client 4 | **63.0%** (8: 98%, 9: 28%) | **83.5%** (4: 90%, 5: 77%) | **60.0%** (12: 88%, 13: 32%) |
