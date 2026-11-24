# Server sync 9 — global model v9

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 43.36% on 36 observed labels.
Validation accuracy: 0.4338888888888889; seen labels never predicted: 0.
Server-accepted classes: 36; pending private classes: 0.
Accepted-only TEST accuracy: 0.4336111111111111.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 | U9 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **60.5%** (0: 79%, 1: 42%) | **42.0%** (4: 46%, 5: 38%) | **42.0%** (8: 36%, 9: 48%) | **56.0%** (12: 56%, 13: 56%) | **63.5%** (16: 52%, 17: 75%) | **68.5%** (20: 75%, 21: 62%) | **49.5%** (24: 56%, 25: 43%) | **49.0%** (28: 45%, 29: 53%) | **16.0%** (32: 8%, 33: 24%) |
| Client 1 | **34.0%** (2: 43%, 3: 25%) | **48.5%** (6: 49%, 7: 48%) | **22.0%** (10: 13%, 11: 31%) | **34.0%** (14: 39%, 15: 29%) | **32.5%** (18: 39%, 19: 26%) | **57.0%** (22: 42%, 23: 72%) | **41.0%** (26: 30%, 27: 52%) | **57.0%** (30: 71%, 31: 43%) | **7.5%** (34: 6%, 35: 9%) |
