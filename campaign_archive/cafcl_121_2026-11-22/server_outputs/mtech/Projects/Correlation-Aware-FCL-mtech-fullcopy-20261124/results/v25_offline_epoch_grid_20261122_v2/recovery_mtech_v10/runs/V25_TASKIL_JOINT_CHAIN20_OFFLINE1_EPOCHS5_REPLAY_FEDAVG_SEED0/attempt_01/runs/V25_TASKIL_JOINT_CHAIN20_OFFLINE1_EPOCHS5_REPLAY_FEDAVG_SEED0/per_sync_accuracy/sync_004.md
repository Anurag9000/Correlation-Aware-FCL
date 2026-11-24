# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 65.14% on 21 observed labels.
Validation accuracy: 0.6466666666666666; seen labels never predicted: 0.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.6604761904761904.
Terminal flush: False. Selected clients: [1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **73.0%** (0: 61%, 1: 85%) | **41.0%** (5: 62%, 6: 20%) | **60.5%** (10: 88%, 11: 33%) | **73.0%** (15: 77%, 16: 69%) |
| Client 1 | **76.0%** (1: 85%, 2: 67%) | **48.0%** (6: 20%, 7: 76%) | **64.0%** (11: 33%, 12: 95%) | **79.5%** (16: 69%, 17: 90%) |
| Client 2 | **77.0%** (2: 67%, 3: 87%) | **85.5%** (7: 76%, 8: 95%) | **49.0%** (12: 95%, 13: 3%) | **91.0%** (17: 90%, 18: 92%) |
| Client 3 | **81.0%** (3: 87%, 4: 75%) | **50.0%** (8: 95%, 9: 5%) | **19.5%** (13: 3%, 14: 36%) | **77.0%** (18: 92%, 19: 62%) |
| Client 4 | **68.5%** (4: 75%, 5: 62%) | **46.5%** (9: 5%, 10: 88%) | **56.5%** (14: 36%, 15: 77%) | **76.0%** (19: 62%, 20: 90%) |
