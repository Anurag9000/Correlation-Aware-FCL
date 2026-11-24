# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 60.58% on 26 observed labels.
Validation accuracy: 0.6153846153846154; seen labels never predicted: 0.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.6292.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **66.5%** (0: 69%, 1: 64%) | **52.5%** (5: 86%, 6: 19%) | **45.0%** (10: 57%, 11: 33%) | **73.5%** (15: 89%, 16: 58%) | **86.0%** (20: 88%, 21: 84%) |
| Client 1 | **69.5%** (1: 64%, 2: 75%) | **49.0%** (6: 19%, 7: 79%) | **63.0%** (11: 33%, 12: 93%) | **79.0%** (16: 58%, 17: 100%) | **48.5%** (21: 84%, 22: 13%) |
| Client 2 | **87.0%** (2: 75%, 3: 99%) | **83.0%** (7: 79%, 8: 87%) | **55.5%** (12: 93%, 13: 18%) | **74.0%** (17: 100%, 18: 48%) | **54.0%** (22: 13%, 23: 95%) |
| Client 3 | **53.5%** (3: 99%, 4: 8%) | **55.5%** (8: 87%, 9: 24%) | **25.0%** (13: 18%, 14: 32%) | **68.0%** (18: 48%, 19: 88%) | **81.5%** (23: 95%, 24: 68%) |
| Client 4 | **47.0%** (4: 8%, 5: 86%) | **40.5%** (9: 24%, 10: 57%) | **60.5%** (14: 32%, 15: 89%) | **88.0%** (19: 88%, 20: 88%) | **34.5%** (24: 68%, 25: 1%) [PENDING upload] |
