# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 42.43% on 60 observed labels.
Validation accuracy: 0.4106666666666667; seen labels never predicted: 2.
Server-accepted classes: 56; pending private classes: 4.
Accepted-only TEST accuracy: 0.43857142857142856.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **43.2%** (0: 18%, 1: 15%, 2: 40%, 3: 100%) | **58.8%** (20: 1%, 21: 80%, 22: 96%, 23: 58%) | **39.2%** (40: 50%, 41: 1%, 42: 98%, 43: 8%) |
| Client 1 | **40.0%** (4: 100%, 5: 39%, 6: 12%, 7: 9%) | **35.8%** (24: 10%, 25: 21%, 26: 12%, 27: 100%) | **22.5%** (44: 0%, 45: 34%, 46: 56%, 47: 0%) [PENDING upload] |
| Client 2 | **34.2%** (8: 99%, 9: 23%, 10: 8%, 11: 7%) | **39.5%** (28: 20%, 29: 0%, 30: 38%, 31: 100%) | **68.8%** (48: 56%, 49: 93%, 50: 72%, 51: 54%) |
| Client 3 | **45.5%** (12: 96%, 13: 35%, 14: 15%, 15: 36%) | **35.0%** (32: 16%, 33: 100%, 34: 6%, 35: 18%) | **36.0%** (52: 3%, 53: 22%, 54: 19%, 55: 100%) |
| Client 4 | **48.0%** (16: 33%, 17: 29%, 18: 37%, 19: 93%) | **49.0%** (36: 3%, 37: 94%, 38: 72%, 39: 27%) | **41.0%** (56: 6%, 57: 36%, 58: 23%, 59: 99%) |
