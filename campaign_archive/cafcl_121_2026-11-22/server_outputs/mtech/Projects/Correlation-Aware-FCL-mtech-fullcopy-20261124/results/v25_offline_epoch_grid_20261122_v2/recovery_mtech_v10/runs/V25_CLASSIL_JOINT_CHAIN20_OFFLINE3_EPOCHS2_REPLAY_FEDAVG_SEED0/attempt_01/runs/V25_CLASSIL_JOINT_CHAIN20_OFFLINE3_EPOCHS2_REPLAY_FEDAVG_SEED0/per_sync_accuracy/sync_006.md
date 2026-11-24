# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 8.58% on 31 observed labels.
Validation accuracy: 0.08903225806451613; seen labels never predicted: 11.
Server-accepted classes: 29; pending private classes: 2.
Accepted-only TEST accuracy: 0.09172413793103448.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **39.0%** (0: 0%, 1: 78%) | **14.0%** (5: 0%, 6: 28%) | **1.5%** (10: 0%, 11: 3%) | **0.0%** (15: 0%, 16: 0%) | **0.0%** (20: 0%, 21: 0%) | **6.0%** (25: 12%, 26: 0%) |
| Client 1 | **43.5%** (1: 78%, 2: 9%) | **15.0%** (6: 28%, 7: 2%) | **11.0%** (11: 3%, 12: 19%) | **0.0%** (16: 0%, 17: 0%) | **0.0%** (21: 0%, 22: 0%) | **0.0%** (26: 0%, 27: 0%) [PENDING upload] |
| Client 2 | **6.0%** (2: 9%, 3: 3%) | **2.0%** (7: 2%, 8: 2%) | **9.5%** (12: 19%, 13: 0%) | **11.0%** (17: 0%, 18: 22%) | **12.0%** (22: 0%, 23: 24%) | **0.0%** (27: 0%, 28: 0%) [PENDING upload] |
| Client 3 | **12.5%** (3: 3%, 4: 22%) | **17.5%** (8: 2%, 9: 33%) | **4.5%** (13: 0%, 14: 9%) | **11.0%** (18: 22%, 19: 0%) | **12.0%** (23: 24%, 24: 0%) | **0.0%** (28: 0%, 29: 0%) [PENDING upload] |
| Client 4 | **11.0%** (4: 22%, 5: 0%) | **16.5%** (9: 33%, 10: 0%) | **4.5%** (14: 9%, 15: 0%) | **0.0%** (19: 0%, 20: 0%) | **6.0%** (24: 0%, 25: 12%) | **0.0%** (29: 0%, 30: 0%) |
