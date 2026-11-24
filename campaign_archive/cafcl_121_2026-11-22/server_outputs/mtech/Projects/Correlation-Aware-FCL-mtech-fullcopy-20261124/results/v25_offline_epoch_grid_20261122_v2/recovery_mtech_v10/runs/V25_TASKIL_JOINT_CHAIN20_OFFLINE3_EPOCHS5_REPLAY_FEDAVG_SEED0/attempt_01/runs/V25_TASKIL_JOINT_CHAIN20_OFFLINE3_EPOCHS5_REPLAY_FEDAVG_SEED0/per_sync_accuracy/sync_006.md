# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 66.74% on 31 observed labels.
Validation accuracy: 0.6793548387096774; seen labels never predicted: 0.
Server-accepted classes: 29; pending private classes: 2.
Accepted-only TEST accuracy: 0.7179310344827586.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **79.5%** (0: 100%, 1: 59%) | **47.0%** (5: 67%, 6: 27%) | **90.5%** (10: 98%, 11: 83%) | **52.0%** (15: 53%, 16: 51%) | **85.5%** (20: 75%, 21: 96%) | **87.5%** (25: 79%, 26: 96%) |
| Client 1 | **53.5%** (1: 59%, 2: 48%) | **54.0%** (6: 27%, 7: 81%) | **78.0%** (11: 83%, 12: 73%) | **75.5%** (16: 51%, 17: 100%) | **82.0%** (21: 96%, 22: 68%) | **51.5%** (26: 96%, 27: 7%) [PENDING upload] |
| Client 2 | **58.0%** (2: 48%, 3: 68%) | **84.5%** (7: 81%, 8: 88%) | **60.0%** (12: 73%, 13: 47%) | **75.0%** (17: 100%, 18: 50%) | **75.5%** (22: 68%, 23: 83%) | **4.5%** (27: 7%, 28: 2%) [PENDING upload] |
| Client 3 | **77.5%** (3: 68%, 4: 87%) | **49.5%** (8: 88%, 9: 11%) | **67.0%** (13: 47%, 14: 87%) | **72.0%** (18: 50%, 19: 94%) | **53.5%** (23: 83%, 24: 24%) | **44.5%** (28: 2%, 29: 87%) [PENDING upload] |
| Client 4 | **77.0%** (4: 87%, 5: 67%) | **54.5%** (9: 11%, 10: 98%) | **70.0%** (14: 87%, 15: 53%) | **84.5%** (19: 94%, 20: 75%) | **51.5%** (24: 24%, 25: 79%) | **83.5%** (29: 87%, 30: 80%) |
