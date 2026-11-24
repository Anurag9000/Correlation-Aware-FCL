# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.05% on 20 observed labels.
Validation accuracy: 0.785; seen labels never predicted: 0.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.7805.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **80.5%** (0: 73%, 1: 88%) | **80.5%** (0: 73%, 1: 88%) | **67.0%** (10: 76%, 11: 58%) | **67.0%** (10: 76%, 11: 58%) |
| Client 1 | **80.5%** (2: 63%, 3: 98%) | **80.5%** (2: 63%, 3: 98%) | **68.5%** (12: 99%, 13: 38%) | **68.5%** (12: 99%, 13: 38%) |
| Client 2 | **81.5%** (4: 88%, 5: 75%) | **81.5%** (4: 88%, 5: 75%) | **76.0%** (14: 53%, 15: 99%) | **76.0%** (14: 53%, 15: 99%) |
| Client 3 | **75.5%** (6: 60%, 7: 91%) | **75.5%** (6: 60%, 7: 91%) | **87.5%** (16: 76%, 17: 99%) | **87.5%** (16: 76%, 17: 99%) |
| Client 4 | **83.5%** (8: 93%, 9: 74%) | **83.5%** (8: 93%, 9: 74%) | **80.0%** (18: 63%, 19: 97%) | **80.0%** (18: 63%, 19: 97%) |
