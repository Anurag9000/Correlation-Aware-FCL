# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 44.65% on 31 observed labels.
Validation accuracy: 0.4735483870967742; seen labels never predicted: 0.
Server-accepted classes: 29; pending private classes: 2.
Accepted-only TEST accuracy: 0.5110344827586207.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **54.5%** (0: 10%, 1: 99%) | **69.5%** (5: 72%, 6: 67%) | **56.5%** (10: 100%, 11: 13%) | **51.5%** (15: 10%, 16: 93%) | **53.0%** (20: 6%, 21: 100%) | **51.0%** (25: 44%, 26: 58%) |
| Client 1 | **56.0%** (1: 99%, 2: 13%) | **65.5%** (6: 67%, 7: 64%) | **12.0%** (11: 13%, 12: 11%) | **84.0%** (16: 93%, 17: 75%) | **56.5%** (21: 100%, 22: 13%) | **36.0%** (26: 58%, 27: 14%) [PENDING upload] |
| Client 2 | **12.0%** (2: 13%, 3: 11%) | **58.0%** (7: 64%, 8: 52%) | **8.5%** (12: 11%, 13: 6%) | **80.5%** (17: 75%, 18: 86%) | **33.5%** (22: 13%, 23: 54%) | **7.0%** (27: 14%, 28: 0%) [PENDING upload] |
| Client 3 | **37.0%** (3: 11%, 4: 63%) | **26.0%** (8: 52%, 9: 0%) | **50.0%** (13: 6%, 14: 94%) | **43.5%** (18: 86%, 19: 1%) | **27.0%** (23: 54%, 24: 0%) | **45.5%** (28: 0%, 29: 91%) [PENDING upload] |
| Client 4 | **67.5%** (4: 63%, 5: 72%) | **50.0%** (9: 0%, 10: 100%) | **52.0%** (14: 94%, 15: 10%) | **3.5%** (19: 1%, 20: 6%) | **22.0%** (24: 0%, 25: 44%) | **77.5%** (29: 91%, 30: 64%) |
