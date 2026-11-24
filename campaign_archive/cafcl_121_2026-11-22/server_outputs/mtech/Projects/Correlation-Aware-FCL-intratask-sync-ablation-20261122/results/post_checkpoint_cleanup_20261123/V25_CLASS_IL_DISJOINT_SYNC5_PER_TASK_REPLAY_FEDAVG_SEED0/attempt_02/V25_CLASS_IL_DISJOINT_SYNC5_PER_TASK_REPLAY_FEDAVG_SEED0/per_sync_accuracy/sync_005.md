# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 51.40% on 10 observed labels.
Validation accuracy: 0.48; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.514.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **71.0%** (0: 79%, 1: 63%) | **71.0%** (0: 79%, 1: 63%) | **71.0%** (0: 79%, 1: 63%) | **71.0%** (0: 79%, 1: 63%) | **71.0%** (0: 79%, 1: 63%) |
| Client 1 | **37.0%** (2: 43%, 3: 31%) | **37.0%** (2: 43%, 3: 31%) | **37.0%** (2: 43%, 3: 31%) | **37.0%** (2: 43%, 3: 31%) | **37.0%** (2: 43%, 3: 31%) |
| Client 2 | **55.5%** (4: 57%, 5: 54%) | **55.5%** (4: 57%, 5: 54%) | **55.5%** (4: 57%, 5: 54%) | **55.5%** (4: 57%, 5: 54%) | **55.5%** (4: 57%, 5: 54%) |
| Client 3 | **49.5%** (6: 38%, 7: 61%) | **49.5%** (6: 38%, 7: 61%) | **49.5%** (6: 38%, 7: 61%) | **49.5%** (6: 38%, 7: 61%) | **49.5%** (6: 38%, 7: 61%) |
| Client 4 | **44.0%** (8: 59%, 9: 29%) | **44.0%** (8: 59%, 9: 29%) | **44.0%** (8: 59%, 9: 29%) | **44.0%** (8: 59%, 9: 29%) | **44.0%** (8: 59%, 9: 29%) |
