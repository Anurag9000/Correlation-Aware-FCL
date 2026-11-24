# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 48.16% on 75 observed labels.
Validation accuracy: 0.4952; seen labels never predicted: 2.
Server-accepted classes: 70; pending private classes: 5.
Accepted-only TEST accuracy: 0.49585714285714283.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **35.6%** (0: 11%, 1: 10%, 2: 30%, 3: 34%, 4: 93%) | **37.6%** (25: 11%, 26: 10%, 27: 99%, 28: 36%, 29: 32%) | **51.6%** (50: 82%, 51: 65%, 52: 91%, 53: 0%, 54: 20%) |
| Client 1 | **57.0%** (5: 73%, 6: 42%, 7: 27%, 8: 95%, 9: 48%) | **58.6%** (30: 94%, 31: 23%, 32: 24%, 33: 92%, 34: 60%) | **28.2%** (55: 1%, 56: 9%, 57: 0%, 58: 40%, 59: 91%) [PENDING upload] |
| Client 2 | **58.0%** (10: 44%, 11: 50%, 12: 52%, 13: 81%, 14: 63%) | **50.0%** (35: 18%, 36: 40%, 37: 77%, 38: 83%, 39: 32%) | **52.4%** (60: 40%, 61: 60%, 62: 38%, 63: 89%, 64: 35%) |
| Client 3 | **47.2%** (15: 17%, 16: 16%, 17: 83%, 18: 99%, 19: 21%) | **42.6%** (40: 22%, 41: 27%, 42: 91%, 43: 19%, 44: 54%) | **50.4%** (65: 48%, 66: 72%, 67: 93%, 68: 35%, 69: 4%) |
| Client 4 | **50.6%** (20: 0%, 21: 93%, 22: 66%, 23: 91%, 24: 3%) | **56.4%** (45: 68%, 46: 35%, 47: 84%, 48: 32%, 49: 63%) | **46.2%** (70: 22%, 71: 2%, 72: 93%, 73: 69%, 74: 45%) |
