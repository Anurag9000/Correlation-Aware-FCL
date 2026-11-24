# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 43.50% on 10 observed labels.
Validation accuracy: 0.396; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.435.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **55.0%** (0: 54%, 1: 56%) | **55.0%** (0: 54%, 1: 56%) | **55.0%** (0: 54%, 1: 56%) | **55.0%** (0: 54%, 1: 56%) | **55.0%** (0: 54%, 1: 56%) |
| Client 1 | **53.5%** (2: 59%, 3: 48%) | **53.5%** (2: 59%, 3: 48%) | **53.5%** (2: 59%, 3: 48%) | **53.5%** (2: 59%, 3: 48%) | **53.5%** (2: 59%, 3: 48%) |
| Client 2 | **28.0%** (4: 45%, 5: 11%) | **28.0%** (4: 45%, 5: 11%) | **28.0%** (4: 45%, 5: 11%) | **28.0%** (4: 45%, 5: 11%) | **28.0%** (4: 45%, 5: 11%) |
| Client 3 | **29.5%** (6: 39%, 7: 20%) | **29.5%** (6: 39%, 7: 20%) | **29.5%** (6: 39%, 7: 20%) | **29.5%** (6: 39%, 7: 20%) | **29.5%** (6: 39%, 7: 20%) |
| Client 4 | **51.5%** (8: 53%, 9: 50%) | **51.5%** (8: 53%, 9: 50%) | **51.5%** (8: 53%, 9: 50%) | **51.5%** (8: 53%, 9: 50%) | **51.5%** (8: 53%, 9: 50%) |
