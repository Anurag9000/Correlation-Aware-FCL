# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 79.40% on 10 observed labels.
Validation accuracy: 0.78; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.794.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **85.5%** (0: 82%, 1: 89%) | **85.5%** (0: 82%, 1: 89%) | **85.5%** (0: 82%, 1: 89%) | **85.5%** (0: 82%, 1: 89%) |
| Client 1 | **84.0%** (2: 68%, 3: 100%) | **84.0%** (2: 68%, 3: 100%) | **84.0%** (2: 68%, 3: 100%) | **84.0%** (2: 68%, 3: 100%) |
| Client 2 | **80.0%** (4: 82%, 5: 78%) | **80.0%** (4: 82%, 5: 78%) | **80.0%** (4: 82%, 5: 78%) | **80.0%** (4: 82%, 5: 78%) |
| Client 3 | **74.0%** (6: 54%, 7: 94%) | **74.0%** (6: 54%, 7: 94%) | **74.0%** (6: 54%, 7: 94%) | **74.0%** (6: 54%, 7: 94%) |
| Client 4 | **73.5%** (8: 95%, 9: 52%) | **73.5%** (8: 95%, 9: 52%) | **73.5%** (8: 95%, 9: 52%) | **73.5%** (8: 95%, 9: 52%) |
