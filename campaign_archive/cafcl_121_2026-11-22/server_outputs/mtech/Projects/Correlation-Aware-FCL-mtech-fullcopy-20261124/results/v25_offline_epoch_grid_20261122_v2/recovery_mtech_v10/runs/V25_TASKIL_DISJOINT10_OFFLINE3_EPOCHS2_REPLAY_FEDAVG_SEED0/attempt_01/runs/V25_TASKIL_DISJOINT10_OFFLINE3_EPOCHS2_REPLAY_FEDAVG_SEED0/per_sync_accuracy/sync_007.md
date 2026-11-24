# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 66.16% on 70 observed labels.
Validation accuracy: 0.6714285714285714; seen labels never predicted: 2.
Server-accepted classes: 52; pending private classes: 18.
Accepted-only TEST accuracy: 0.7267307692307692.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **76.5%** (0: 74%, 1: 79%) | **54.0%** (10: 87%, 11: 21%) | **83.5%** (20: 67%, 21: 100%) | **87.5%** (30: 89%, 31: 86%) | **58.0%** (40: 19%, 41: 97%) | **59.5%** (50: 93%, 51: 26%) | **88.5%** (60: 90%, 61: 87%) |
| Client 1 | **87.5%** (2: 82%, 3: 93%) | **61.5%** (12: 85%, 13: 38%) | **82.0%** (22: 75%, 23: 89%) | **48.5%** (32: 97%, 33: 0%) [PENDING upload] | **49.0%** (42: 98%, 43: 0%) [PENDING upload] | **50.0%** (52: 100%, 53: 0%) [PENDING upload] | **61.5%** (62: 99%, 63: 24%) [PENDING upload] |
| Client 2 | **73.5%** (4: 87%, 5: 60%) | **68.5%** (14: 64%, 15: 73%) | **71.0%** (24: 70%, 25: 72%) | **71.5%** (34: 88%, 35: 55%) | **62.0%** (44: 96%, 45: 28%) | **44.5%** (54: 89%, 55: 0%) [PENDING upload] | **51.5%** (64: 95%, 65: 8%) [PENDING upload] |
| Client 3 | **62.0%** (6: 40%, 7: 84%) | **73.5%** (16: 49%, 17: 98%) | **74.0%** (26: 52%, 27: 96%) | **71.5%** (36: 48%, 37: 95%) | **50.0%** (46: 100%, 47: 0%) [PENDING upload] | **35.5%** (56: 35%, 57: 36%) [PENDING upload] | **35.5%** (66: 38%, 67: 33%) [PENDING upload] |
| Client 4 | **70.0%** (8: 92%, 9: 48%) | **68.5%** (18: 67%, 19: 70%) | **72.0%** (28: 62%, 29: 82%) | **68.0%** (38: 90%, 39: 46%) | **87.0%** (48: 81%, 49: 93%) | **67.5%** (58: 51%, 59: 84%) | **90.5%** (68: 97%, 69: 84%) |
