# Server sync 8 — global model v8

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 19.15% on 80 observed labels.
Validation accuracy: 0.18975; seen labels never predicted: 29.
Server-accepted classes: 62; pending private classes: 18.
Accepted-only TEST accuracy: 0.2470967741935484.
Terminal flush: False. Selected clients: [3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 | U8 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **5.5%** (0: 7%, 1: 4%) | **5.5%** (10: 11%, 11: 0%) | **37.5%** (20: 43%, 21: 32%) | **47.5%** (30: 66%, 31: 29%) | **18.5%** (40: 4%, 41: 33%) | **4.0%** (50: 0%, 51: 8%) | **18.0%** (60: 32%, 61: 4%) | **0.0%** (70: 0%, 71: 0%) [PENDING upload] |
| Client 1 | **20.5%** (2: 20%, 3: 21%) | **57.0%** (12: 40%, 13: 74%) | **50.5%** (22: 25%, 23: 76%) | **0.0%** (32: 0%, 33: 0%) [PENDING upload] | **0.0%** (42: 0%, 43: 0%) [PENDING upload] | **0.0%** (52: 0%, 53: 0%) [PENDING upload] | **0.0%** (62: 0%, 63: 0%) [PENDING upload] | **0.0%** (72: 0%, 73: 0%) [PENDING upload] |
| Client 2 | **35.0%** (4: 52%, 5: 18%) | **40.5%** (14: 46%, 15: 35%) | **15.5%** (24: 30%, 25: 1%) | **13.0%** (34: 20%, 35: 6%) | **6.5%** (44: 12%, 45: 1%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] | **0.0%** (64: 0%, 65: 0%) [PENDING upload] | **0.0%** (74: 0%, 75: 0%) [PENDING upload] |
| Client 3 | **51.5%** (6: 39%, 7: 64%) | **38.0%** (16: 28%, 17: 48%) | **39.5%** (26: 30%, 27: 49%) | **29.0%** (36: 27%, 37: 31%) | **0.0%** (46: 0%, 47: 0%) | **0.0%** (56: 0%, 57: 0%) | **0.0%** (66: 0%, 67: 0%) | **0.0%** (76: 0%, 77: 0%) |
| Client 4 | **55.5%** (8: 72%, 9: 39%) | **30.5%** (18: 26%, 19: 35%) | **31.5%** (28: 35%, 29: 28%) | **25.0%** (38: 7%, 39: 43%) | **45.0%** (48: 59%, 49: 31%) | **23.0%** (58: 3%, 59: 43%) | **22.5%** (68: 25%, 69: 20%) | **0.0%** (78: 0%, 79: 0%) |
