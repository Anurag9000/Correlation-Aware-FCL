# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 72.19% on 26 observed labels.
Validation accuracy: 0.7330769230769231; seen labels never predicted: 0.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.7407692307692307.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **77.0%** (0: 70%, 1: 84%) | **58.0%** (5: 87%, 6: 29%) | **85.5%** (10: 88%, 11: 83%) | **90.0%** (15: 90%, 16: 90%) | **90.0%** (20: 82%, 21: 98%) |
| Client 1 | **81.5%** (1: 84%, 2: 79%) | **57.5%** (6: 29%, 7: 86%) | **83.5%** (11: 83%, 12: 84%) | **89.5%** (16: 90%, 17: 89%) | **61.5%** (21: 98%, 22: 25%) |
| Client 2 | **89.5%** (2: 79%, 3: 100%) | **91.0%** (7: 86%, 8: 96%) | **87.0%** (12: 84%, 13: 90%) | **85.0%** (17: 89%, 18: 81%) | **62.0%** (22: 25%, 23: 99%) |
| Client 3 | **50.0%** (3: 100%, 4: 0%) | **59.5%** (8: 96%, 9: 23%) | **55.5%** (13: 90%, 14: 21%) | **65.0%** (18: 81%, 19: 49%) | **87.5%** (23: 99%, 24: 76%) |
| Client 4 | **43.5%** (4: 0%, 5: 87%) | **55.5%** (9: 23%, 10: 88%) | **55.5%** (14: 21%, 15: 90%) | **65.5%** (19: 49%, 20: 82%) | **77.0%** (24: 76%, 25: 78%) |
