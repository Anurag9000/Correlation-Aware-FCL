# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 35.60% on 10 observed labels.
Validation accuracy: 0.314; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.356.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **52.5%** (0: 56%, 1: 49%) | **52.5%** (0: 56%, 1: 49%) | **52.5%** (0: 56%, 1: 49%) | **52.5%** (0: 56%, 1: 49%) |
| Client 1 | **23.0%** (2: 25%, 3: 21%) | **23.0%** (2: 25%, 3: 21%) | **23.0%** (2: 25%, 3: 21%) | **23.0%** (2: 25%, 3: 21%) |
| Client 2 | **43.0%** (4: 58%, 5: 28%) | **43.0%** (4: 58%, 5: 28%) | **43.0%** (4: 58%, 5: 28%) | **43.0%** (4: 58%, 5: 28%) |
| Client 3 | **18.5%** (6: 27%, 7: 10%) | **18.5%** (6: 27%, 7: 10%) | **18.5%** (6: 27%, 7: 10%) | **18.5%** (6: 27%, 7: 10%) |
| Client 4 | **41.0%** (8: 72%, 9: 10%) | **41.0%** (8: 72%, 9: 10%) | **41.0%** (8: 72%, 9: 10%) | **41.0%** (8: 72%, 9: 10%) |
