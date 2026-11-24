# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 66.31% on 36 observed labels.
Validation accuracy: 0.6588888888888889; seen labels never predicted: 0.
Server-accepted classes: 32; pending private classes: 4.
Accepted-only TEST accuracy: 0.7590625.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **76.5%** (0: 99%, 1: 54%) | **70.0%** (5: 88%, 6: 52%) | **68.0%** (10: 72%, 11: 64%) | **75.5%** (15: 85%, 16: 66%) | **93.5%** (20: 87%, 21: 100%) | **76.5%** (25: 54%, 26: 99%) | **83.5%** (30: 100%, 31: 67%) |
| Client 1 | **65.0%** (1: 54%, 2: 76%) | **58.5%** (6: 52%, 7: 65%) | **73.0%** (11: 64%, 12: 82%) | **78.5%** (16: 66%, 17: 91%) | **61.5%** (21: 100%, 22: 23%) | **49.5%** (26: 99%, 27: 0%) [PENDING upload] | **34.5%** (31: 67%, 32: 2%) [PENDING upload] |
| Client 2 | **68.0%** (2: 76%, 3: 60%) | **66.5%** (7: 65%, 8: 68%) | **71.5%** (12: 82%, 13: 61%) | **83.0%** (17: 91%, 18: 75%) | **58.5%** (22: 23%, 23: 94%) | **10.0%** (27: 0%, 28: 20%) [PENDING upload] | **30.0%** (32: 2%, 33: 58%) [PENDING upload] |
| Client 3 | **71.0%** (3: 60%, 4: 82%) | **67.0%** (8: 68%, 9: 66%) | **62.5%** (13: 61%, 14: 64%) | **76.0%** (18: 75%, 19: 77%) | **51.0%** (23: 94%, 24: 8%) | **45.0%** (28: 20%, 29: 70%) [PENDING upload] | **73.5%** (33: 58%, 34: 89%) [PENDING upload] |
| Client 4 | **85.0%** (4: 82%, 5: 88%) | **69.0%** (9: 66%, 10: 72%) | **74.5%** (14: 64%, 15: 85%) | **82.0%** (19: 77%, 20: 87%) | **31.0%** (24: 8%, 25: 54%) | **85.0%** (29: 70%, 30: 100%) | **79.0%** (34: 89%, 35: 69%) |
