# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 16.23% on 26 observed labels.
Validation accuracy: 0.16230769230769232; seen labels never predicted: 6.
Server-accepted classes: 23; pending private classes: 3.
Accepted-only TEST accuracy: 0.18347826086956523.
Terminal flush: False. Selected clients: [2, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **42.5%** (0: 0%, 1: 85%) [PENDING upload] | **26.0%** (5: 35%, 6: 17%) | **3.5%** (10: 6%, 11: 1%) | **0.0%** (15: 0%, 16: 0%) [PENDING upload] | **1.0%** (20: 2%, 21: 0%) [PENDING upload] |
| Client 1 | **56.5%** (1: 85%, 2: 28%) | **40.5%** (6: 17%, 7: 64%) | **26.0%** (11: 1%, 12: 51%) | **0.0%** (16: 0%, 17: 0%) [PENDING upload] | **0.0%** (21: 0%, 22: 0%) [PENDING upload] |
| Client 2 | **17.5%** (2: 28%, 3: 7%) | **37.0%** (7: 64%, 8: 10%) | **27.5%** (12: 51%, 13: 4%) | **11.0%** (17: 0%, 18: 22%) | **0.5%** (22: 0%, 23: 1%) |
| Client 3 | **38.5%** (3: 7%, 4: 70%) | **6.0%** (8: 10%, 9: 2%) | **2.0%** (13: 4%, 14: 0%) | **19.5%** (18: 22%, 19: 17%) | **0.5%** (23: 1%, 24: 0%) |
| Client 4 | **52.5%** (4: 70%, 5: 35%) | **4.0%** (9: 2%, 10: 6%) | **0.0%** (14: 0%, 15: 0%) | **9.5%** (19: 17%, 20: 2%) | **0.0%** (24: 0%, 25: 0%) |
