# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 82.88% on 26 observed labels.
Validation accuracy: 0.8430769230769231; seen labels never predicted: 0.
Server-accepted classes: 26; pending private classes: 0.
Accepted-only TEST accuracy: 0.8288461538461539.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **80.5%** (0: 74%, 1: 87%) | **71.0%** (10: 75%, 11: 67%) | **80.0%** (14: 69%, 15: 91%) | **71.0%** (10: 75%, 11: 67%) | **84.0%** (22: 79%, 23: 89%) |
| Client 1 | **86.0%** (2: 82%, 3: 90%) | **85.5%** (4: 86%, 5: 85%) | **86.0%** (2: 82%, 3: 90%) | **92.0%** (16: 89%, 17: 95%) | **71.0%** (10: 75%, 11: 67%) |
| Client 2 | **85.5%** (4: 86%, 5: 85%) | **84.0%** (8: 94%, 9: 74%) | **78.5%** (12: 88%, 13: 69%) | **79.0%** (18: 82%, 19: 76%) | **86.0%** (2: 82%, 3: 90%) |
| Client 3 | **77.5%** (6: 68%, 7: 87%) | **78.5%** (12: 88%, 13: 69%) | **92.0%** (16: 89%, 17: 95%) | **92.5%** (20: 85%, 21: 100%) | **87.0%** (24: 93%, 25: 81%) |
| Client 4 | **84.0%** (8: 94%, 9: 74%) | **86.0%** (2: 82%, 3: 90%) | **71.0%** (10: 75%, 11: 67%) | **86.0%** (2: 82%, 3: 90%) | **92.5%** (20: 85%, 21: 100%) |
