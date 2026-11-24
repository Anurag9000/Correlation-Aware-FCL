# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 31.90% on 10 observed labels.
Validation accuracy: 0.27; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.319.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **37.0%** (0: 48%, 1: 26%) | **37.0%** (0: 48%, 1: 26%) | **37.0%** (0: 48%, 1: 26%) |
| Client 1 | **32.0%** (2: 37%, 3: 27%) | **32.0%** (2: 37%, 3: 27%) | **32.0%** (2: 37%, 3: 27%) |
| Client 2 | **37.0%** (4: 63%, 5: 11%) | **37.0%** (4: 63%, 5: 11%) | **37.0%** (4: 63%, 5: 11%) |
| Client 3 | **14.0%** (6: 26%, 7: 2%) | **14.0%** (6: 26%, 7: 2%) | **14.0%** (6: 26%, 7: 2%) |
| Client 4 | **39.5%** (8: 72%, 9: 7%) | **39.5%** (8: 72%, 9: 7%) | **39.5%** (8: 72%, 9: 7%) |
