# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 18.13% on 60 observed labels.
Validation accuracy: 0.18133333333333335; seen labels never predicted: 15.
Server-accepted classes: 58; pending private classes: 2.
Accepted-only TEST accuracy: 0.18758620689655173.
Terminal flush: False. Selected clients: [0, 1, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **49.5%** (0: 42%, 1: 57%) | **4.0%** (10: 1%, 11: 7%) | **43.0%** (20: 43%, 21: 43%) | **13.5%** (30: 27%, 31: 0%) | **0.5%** (40: 0%, 41: 1%) | **0.0%** (50: 0%, 51: 0%) |
| Client 1 | **25.5%** (2: 15%, 3: 36%) | **48.0%** (12: 56%, 13: 40%) | **37.5%** (22: 4%, 23: 71%) | **24.5%** (32: 1%, 33: 48%) | **0.5%** (42: 1%, 43: 0%) | **0.0%** (52: 0%, 53: 0%) |
| Client 2 | **30.5%** (4: 45%, 5: 16%) | **21.0%** (14: 16%, 15: 26%) | **33.5%** (24: 60%, 25: 7%) | **5.0%** (34: 6%, 35: 4%) | **0.0%** (44: 0%, 45: 0%) | **0.0%** (54: 0%, 55: 0%) [PENDING upload] |
| Client 3 | **27.0%** (6: 28%, 7: 26%) | **39.0%** (16: 15%, 17: 63%) | **30.0%** (26: 6%, 27: 54%) | **31.5%** (36: 29%, 37: 34%) | **0.5%** (46: 0%, 47: 1%) | **0.0%** (56: 0%, 57: 0%) |
| Client 4 | **22.0%** (8: 23%, 9: 21%) | **20.0%** (18: 13%, 19: 27%) | **23.5%** (28: 18%, 29: 29%) | **14.0%** (38: 22%, 39: 6%) | **0.0%** (48: 0%, 49: 0%) | **0.0%** (58: 0%, 59: 0%) |
