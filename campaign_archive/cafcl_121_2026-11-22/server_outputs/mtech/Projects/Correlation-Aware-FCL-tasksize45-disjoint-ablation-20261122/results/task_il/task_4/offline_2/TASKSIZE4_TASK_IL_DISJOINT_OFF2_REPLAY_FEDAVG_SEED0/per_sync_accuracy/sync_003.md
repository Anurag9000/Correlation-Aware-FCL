# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 58.53% on 60 observed labels.
Validation accuracy: 0.5843333333333334; seen labels never predicted: 2.
Server-accepted classes: 40; pending private classes: 20.
Accepted-only TEST accuracy: 0.74575.
Terminal flush: False. Selected clients: [1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **81.5%** (0: 80%, 1: 65%, 2: 89%, 3: 92%) | **26.2%** (20: 98%, 21: 2%, 22: 0%, 23: 5%) [PENDING upload] | **20.5%** (40: 3%, 41: 12%, 42: 67%, 43: 0%) [PENDING upload] |
| Client 1 | **74.0%** (4: 81%, 5: 88%, 6: 72%, 7: 55%) | **73.5%** (24: 76%, 25: 55%, 26: 69%, 27: 94%) | **70.2%** (44: 81%, 45: 57%, 46: 65%, 47: 78%) |
| Client 2 | **25.8%** (8: 0%, 9: 10%, 10: 25%, 11: 68%) [PENDING upload] | **34.0%** (28: 96%, 29: 15%, 30: 0%, 31: 25%) [PENDING upload] | **25.8%** (48: 2%, 49: 0%, 50: 26%, 51: 75%) [PENDING upload] |
| Client 3 | **72.2%** (12: 81%, 13: 50%, 14: 74%, 15: 84%) | **68.5%** (32: 56%, 33: 93%, 34: 56%, 35: 69%) | **84.0%** (52: 98%, 53: 70%, 54: 70%, 55: 98%) |
| Client 4 | **76.0%** (16: 69%, 17: 92%, 18: 79%, 19: 64%) | **72.8%** (36: 63%, 37: 82%, 38: 85%, 39: 61%) | **73.0%** (56: 62%, 57: 85%, 58: 65%, 59: 80%) |
