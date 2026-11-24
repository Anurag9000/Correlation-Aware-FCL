# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 61.81% on 21 observed labels.
Validation accuracy: 0.6390476190476191; seen labels never predicted: 0.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.6238095238095238.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **74.5%** (0: 94%, 1: 55%) | **61.0%** (5: 95%, 6: 27%) | **63.5%** (10: 75%, 11: 52%) | **89.5%** (15: 87%, 16: 92%) |
| Client 1 | **64.5%** (1: 55%, 2: 74%) | **57.5%** (6: 27%, 7: 88%) | **74.0%** (11: 52%, 12: 96%) | **85.5%** (16: 92%, 17: 79%) |
| Client 2 | **84.5%** (2: 74%, 3: 95%) | **67.5%** (7: 88%, 8: 47%) | **50.5%** (12: 96%, 13: 5%) | **55.5%** (17: 79%, 18: 32%) |
| Client 3 | **47.5%** (3: 95%, 4: 0%) | **28.5%** (8: 47%, 9: 10%) | **19.5%** (13: 5%, 14: 34%) | **50.0%** (18: 32%, 19: 68%) |
| Client 4 | **47.5%** (4: 0%, 5: 95%) | **42.5%** (9: 10%, 10: 75%) | **60.5%** (14: 34%, 15: 87%) | **80.5%** (19: 68%, 20: 93%) |
