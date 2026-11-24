# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.62% on 60 observed labels.
Validation accuracy: 0.7916666666666666; seen labels never predicted: 0.
Server-accepted classes: 48; pending private classes: 12.
Accepted-only TEST accuracy: 0.8675.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **85.0%** (8: 89%, 9: 81%) | **78.0%** (14: 71%, 15: 85%) | **84.5%** (26: 85%, 27: 84%) | **80.5%** (34: 71%, 35: 90%) | **93.0%** (48: 90%, 49: 96%) | **85.5%** (56: 97%, 57: 74%) |
| Client 1 | **50.5%** (6: 3%, 7: 98%) [PENDING upload] | **57.5%** (18: 16%, 19: 99%) [PENDING upload] | **46.0%** (28: 37%, 29: 55%) [PENDING upload] | **27.0%** (30: 54%, 31: 0%) [PENDING upload] | **64.0%** (42: 63%, 43: 65%) [PENDING upload] | **31.5%** (54: 10%, 55: 53%) [PENDING upload] |
| Client 2 | **88.5%** (4: 93%, 5: 84%) | **95.0%** (16: 94%, 17: 96%) | **91.0%** (22: 88%, 23: 94%) | **82.5%** (32: 80%, 33: 85%) | **92.0%** (46: 88%, 47: 96%) | **86.0%** (58: 74%, 59: 98%) |
| Client 3 | **90.5%** (0: 95%, 1: 86%) | **80.5%** (10: 86%, 11: 75%) | **94.0%** (20: 92%, 21: 96%) | **88.0%** (38: 89%, 39: 87%) | **75.5%** (44: 82%, 45: 69%) | **78.5%** (50: 67%, 51: 90%) |
| Client 4 | **88.0%** (2: 90%, 3: 86%) | **75.5%** (12: 84%, 13: 67%) | **90.5%** (24: 92%, 25: 89%) | **92.5%** (36: 95%, 37: 90%) | **88.5%** (40: 88%, 41: 89%) | **98.5%** (52: 100%, 53: 97%) |
