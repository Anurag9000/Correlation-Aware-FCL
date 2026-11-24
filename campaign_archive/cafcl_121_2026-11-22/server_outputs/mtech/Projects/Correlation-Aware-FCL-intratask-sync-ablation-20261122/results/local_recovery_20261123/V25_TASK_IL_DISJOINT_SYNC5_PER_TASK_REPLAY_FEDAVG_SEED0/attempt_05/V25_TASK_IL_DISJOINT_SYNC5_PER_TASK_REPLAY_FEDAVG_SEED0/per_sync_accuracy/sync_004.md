# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 78.30% on 10 observed labels.
Validation accuracy: 0.758; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.783.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **84.5%** (0: 94%, 1: 75%) | **84.5%** (0: 94%, 1: 75%) | **84.5%** (0: 94%, 1: 75%) | **84.5%** (0: 94%, 1: 75%) |
| Client 1 | **79.5%** (2: 60%, 3: 99%) | **79.5%** (2: 60%, 3: 99%) | **79.5%** (2: 60%, 3: 99%) | **79.5%** (2: 60%, 3: 99%) |
| Client 2 | **79.0%** (4: 84%, 5: 74%) | **79.0%** (4: 84%, 5: 74%) | **79.0%** (4: 84%, 5: 74%) | **79.0%** (4: 84%, 5: 74%) |
| Client 3 | **74.0%** (6: 58%, 7: 90%) | **74.0%** (6: 58%, 7: 90%) | **74.0%** (6: 58%, 7: 90%) | **74.0%** (6: 58%, 7: 90%) |
| Client 4 | **74.5%** (8: 93%, 9: 56%) | **74.5%** (8: 93%, 9: 56%) | **74.5%** (8: 93%, 9: 56%) | **74.5%** (8: 93%, 9: 56%) |
