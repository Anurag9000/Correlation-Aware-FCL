# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.90% on 21 observed labels.
Validation accuracy: 0.6295238095238095; seen labels never predicted: 0.
Server-accepted classes: 18; pending private classes: 3.
Accepted-only TEST accuracy: 0.645.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **78.5%** (0: 69%, 1: 88%) [PENDING upload] | **76.5%** (5: 54%, 6: 99%) | **74.5%** (10: 56%, 11: 93%) | **57.0%** (15: 21%, 16: 93%) [PENDING upload] |
| Client 1 | **83.5%** (1: 88%, 2: 79%) | **49.5%** (6: 99%, 7: 0%) | **77.0%** (11: 93%, 12: 61%) | **78.5%** (16: 93%, 17: 64%) [PENDING upload] |
| Client 2 | **65.0%** (2: 79%, 3: 51%) | **20.5%** (7: 0%, 8: 41%) | **34.0%** (12: 61%, 13: 7%) | **56.0%** (17: 64%, 18: 48%) [PENDING upload] |
| Client 3 | **72.0%** (3: 51%, 4: 93%) | **53.0%** (8: 41%, 9: 65%) | **49.5%** (13: 7%, 14: 92%) | **42.5%** (18: 48%, 19: 37%) |
| Client 4 | **73.5%** (4: 93%, 5: 54%) | **60.5%** (9: 65%, 10: 56%) | **56.5%** (14: 92%, 15: 21%) | **63.0%** (19: 37%, 20: 89%) |
