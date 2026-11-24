# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 83.88% on 34 observed labels.
Validation accuracy: 0.8494117647058823; seen labels never predicted: 0.
Server-accepted classes: 34; pending private classes: 0.
Accepted-only TEST accuracy: 0.8388235294117647.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **85.5%** (0: 81%, 1: 90%) | **85.0%** (8: 88%, 9: 82%) | **94.5%** (16: 91%, 17: 98%) | **82.5%** (22: 70%, 23: 95%) | **96.0%** (20: 92%, 21: 100%) |
| Client 1 | **88.5%** (2: 92%, 3: 85%) | **71.0%** (10: 83%, 11: 59%) | **85.0%** (8: 88%, 9: 82%) | **94.5%** (16: 91%, 17: 98%) | **79.0%** (28: 64%, 29: 94%) |
| Client 2 | **87.0%** (4: 87%, 5: 87%) | **75.0%** (12: 92%, 13: 58%) | **83.0%** (18: 94%, 19: 72%) | **96.0%** (20: 92%, 21: 100%) | **94.0%** (30: 99%, 31: 89%) |
| Client 3 | **79.0%** (6: 66%, 7: 92%) | **80.5%** (14: 77%, 15: 84%) | **96.0%** (20: 92%, 21: 100%) | **87.0%** (24: 92%, 25: 82%) | **82.5%** (22: 70%, 23: 95%) |
| Client 4 | **85.0%** (8: 88%, 9: 82%) | **87.0%** (4: 87%, 5: 87%) | **75.0%** (12: 92%, 13: 58%) | **75.5%** (26: 59%, 27: 92%) | **83.0%** (32: 85%, 33: 81%) |
