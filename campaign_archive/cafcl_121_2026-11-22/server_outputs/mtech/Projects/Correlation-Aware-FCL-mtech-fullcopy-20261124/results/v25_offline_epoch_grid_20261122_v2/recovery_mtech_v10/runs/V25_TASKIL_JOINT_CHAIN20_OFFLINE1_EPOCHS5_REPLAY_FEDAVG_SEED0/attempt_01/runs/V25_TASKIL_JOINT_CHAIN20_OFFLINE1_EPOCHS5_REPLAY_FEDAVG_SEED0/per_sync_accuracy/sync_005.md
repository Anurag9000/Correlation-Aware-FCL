# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 67.54% on 26 observed labels.
Validation accuracy: 0.6661538461538462; seen labels never predicted: 0.
Server-accepted classes: 25; pending private classes: 1.
Accepted-only TEST accuracy: 0.7012.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **72.0%** (0: 58%, 1: 86%) | **55.0%** (5: 69%, 6: 41%) | **67.5%** (10: 86%, 11: 49%) | **62.0%** (15: 36%, 16: 88%) | **90.5%** (20: 85%, 21: 96%) |
| Client 1 | **76.0%** (1: 86%, 2: 66%) | **54.0%** (6: 41%, 7: 67%) | **72.0%** (11: 49%, 12: 95%) | **91.5%** (16: 88%, 17: 95%) | **66.0%** (21: 96%, 22: 36%) |
| Client 2 | **81.0%** (2: 66%, 3: 96%) | **80.5%** (7: 67%, 8: 94%) | **67.5%** (12: 95%, 13: 40%) | **89.0%** (17: 95%, 18: 83%) | **66.5%** (22: 36%, 23: 97%) |
| Client 3 | **82.0%** (3: 96%, 4: 68%) | **61.5%** (8: 94%, 9: 29%) | **43.0%** (13: 40%, 14: 46%) | **79.5%** (18: 83%, 19: 76%) | **84.5%** (23: 97%, 24: 72%) |
| Client 4 | **68.5%** (4: 68%, 5: 69%) | **57.5%** (9: 29%, 10: 86%) | **41.0%** (14: 46%, 15: 36%) | **80.5%** (19: 76%, 20: 85%) | **37.0%** (24: 72%, 25: 2%) [PENDING upload] |
