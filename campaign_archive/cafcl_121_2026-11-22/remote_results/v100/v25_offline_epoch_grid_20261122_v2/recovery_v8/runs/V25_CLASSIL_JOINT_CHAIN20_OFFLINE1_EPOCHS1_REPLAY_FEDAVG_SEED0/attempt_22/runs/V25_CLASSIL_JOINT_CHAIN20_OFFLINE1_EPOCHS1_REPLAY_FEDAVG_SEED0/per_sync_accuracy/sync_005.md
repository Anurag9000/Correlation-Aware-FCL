# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.62% on 26 observed labels.
Validation accuracy: 0.1776923076923077; seen labels never predicted: 8.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.1728.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **66.5%** (0: 70%, 1: 63%) | **16.5%** (5: 12%, 6: 21%) | **1.0%** (10: 0%, 11: 2%) | **0.0%** (15: 0%, 16: 0%) | **53.5%** (20: 66%, 21: 41%) |
| Client 1 | **42.0%** (1: 63%, 2: 21%) | **10.5%** (6: 21%, 7: 0%) | **36.5%** (11: 2%, 12: 71%) | **0.0%** (16: 0%, 17: 0%) | **24.5%** (21: 41%, 22: 8%) |
| Client 2 | **11.0%** (2: 21%, 3: 1%) | **7.0%** (7: 0%, 8: 14%) | **35.5%** (12: 71%, 13: 0%) | **4.5%** (17: 0%, 18: 9%) | **6.0%** (22: 8%, 23: 4%) |
| Client 3 | **12.0%** (3: 1%, 4: 23%) | **7.0%** (8: 14%, 9: 0%) | **3.0%** (13: 0%, 14: 6%) | **4.5%** (18: 9%, 19: 0%) | **2.0%** (23: 4%, 24: 0%) |
| Client 4 | **17.5%** (4: 23%, 5: 12%) | **0.0%** (9: 0%, 10: 0%) | **3.0%** (14: 6%, 15: 0%) | **33.0%** (19: 0%, 20: 66%) | **0.0%** (24: 0%, 25: 0%) [PENDING upload] |
