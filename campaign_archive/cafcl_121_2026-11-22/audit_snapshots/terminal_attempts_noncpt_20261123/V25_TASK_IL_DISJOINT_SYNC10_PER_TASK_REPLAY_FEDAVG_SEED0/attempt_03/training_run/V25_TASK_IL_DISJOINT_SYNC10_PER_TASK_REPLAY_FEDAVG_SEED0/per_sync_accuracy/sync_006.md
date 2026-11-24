# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 79.90% on 10 observed labels.
Validation accuracy: 0.8; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.799.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **88.5%** (0: 85%, 1: 92%) | **88.5%** (0: 85%, 1: 92%) | **88.5%** (0: 85%, 1: 92%) | **88.5%** (0: 85%, 1: 92%) | **88.5%** (0: 85%, 1: 92%) | **88.5%** (0: 85%, 1: 92%) |
| Client 1 | **88.0%** (2: 90%, 3: 86%) | **88.0%** (2: 90%, 3: 86%) | **88.0%** (2: 90%, 3: 86%) | **88.0%** (2: 90%, 3: 86%) | **88.0%** (2: 90%, 3: 86%) | **88.0%** (2: 90%, 3: 86%) |
| Client 2 | **79.0%** (4: 87%, 5: 71%) | **79.0%** (4: 87%, 5: 71%) | **79.0%** (4: 87%, 5: 71%) | **79.0%** (4: 87%, 5: 71%) | **79.0%** (4: 87%, 5: 71%) | **79.0%** (4: 87%, 5: 71%) |
| Client 3 | **73.0%** (6: 60%, 7: 86%) | **73.0%** (6: 60%, 7: 86%) | **73.0%** (6: 60%, 7: 86%) | **73.0%** (6: 60%, 7: 86%) | **73.0%** (6: 60%, 7: 86%) | **73.0%** (6: 60%, 7: 86%) |
| Client 4 | **71.0%** (8: 81%, 9: 61%) | **71.0%** (8: 81%, 9: 61%) | **71.0%** (8: 81%, 9: 61%) | **71.0%** (8: 81%, 9: 61%) | **71.0%** (8: 81%, 9: 61%) | **71.0%** (8: 81%, 9: 61%) |
