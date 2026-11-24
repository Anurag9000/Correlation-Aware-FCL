# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 75.80% on 50 observed labels.
Validation accuracy: 0.7748; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 2.
Accepted-only TEST accuracy: 0.776875.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **86.0%** (8: 83%, 9: 89%) | **76.5%** (14: 68%, 15: 85%) | **68.5%** (26: 39%, 27: 98%) | **78.0%** (34: 86%, 35: 70%) | **91.0%** (48: 84%, 49: 98%) |
| Client 1 | **68.5%** (6: 72%, 7: 65%) | **75.5%** (18: 77%, 19: 74%) | **79.0%** (28: 66%, 29: 92%) | **93.0%** (30: 89%, 31: 97%) | **62.0%** (42: 85%, 43: 39%) |
| Client 2 | **74.0%** (4: 97%, 5: 51%) | **92.5%** (16: 86%, 17: 99%) | **88.0%** (22: 81%, 23: 95%) | **74.0%** (32: 54%, 33: 94%) | **30.5%** (46: 25%, 47: 36%) [PENDING upload] |
| Client 3 | **86.5%** (0: 83%, 1: 90%) | **75.5%** (10: 90%, 11: 61%) | **91.5%** (20: 84%, 21: 99%) | **80.0%** (38: 95%, 39: 65%) | **59.5%** (44: 99%, 45: 20%) |
| Client 4 | **82.5%** (2: 68%, 3: 97%) | **70.5%** (12: 82%, 13: 59%) | **74.5%** (24: 98%, 25: 51%) | **85.5%** (36: 83%, 37: 88%) | **52.0%** (40: 97%, 41: 7%) |
