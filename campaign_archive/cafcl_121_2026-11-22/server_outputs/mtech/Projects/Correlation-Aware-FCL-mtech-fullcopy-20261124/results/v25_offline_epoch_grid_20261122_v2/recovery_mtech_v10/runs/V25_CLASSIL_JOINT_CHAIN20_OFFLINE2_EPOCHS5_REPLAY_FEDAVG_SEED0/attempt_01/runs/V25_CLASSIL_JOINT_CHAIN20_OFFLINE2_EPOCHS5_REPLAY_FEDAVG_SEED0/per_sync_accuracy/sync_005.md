# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 25.77% on 26 observed labels.
Validation accuracy: 0.23461538461538461; seen labels never predicted: 4.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.25769230769230766.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **62.5%** (0: 56%, 1: 69%) | **53.0%** (5: 65%, 6: 41%) | **4.5%** (10: 3%, 11: 6%) | **24.0%** (15: 48%, 16: 0%) | **3.0%** (20: 6%, 21: 0%) |
| Client 1 | **53.0%** (1: 69%, 2: 37%) | **44.0%** (6: 41%, 7: 47%) | **26.0%** (11: 6%, 12: 46%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) |
| Client 2 | **48.0%** (2: 37%, 3: 59%) | **33.0%** (7: 47%, 8: 19%) | **45.5%** (12: 46%, 13: 45%) | **10.0%** (17: 0%, 18: 20%) | **5.5%** (22: 0%, 23: 11%) |
| Client 3 | **34.0%** (3: 59%, 4: 9%) | **24.5%** (8: 19%, 9: 30%) | **24.5%** (13: 45%, 14: 4%) | **10.5%** (18: 20%, 19: 1%) | **29.5%** (23: 11%, 24: 48%) |
| Client 4 | **37.0%** (4: 9%, 5: 65%) | **16.5%** (9: 30%, 10: 3%) | **26.0%** (14: 4%, 15: 48%) | **3.5%** (19: 1%, 20: 6%) | **24.0%** (24: 48%, 25: 0%) |
