# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 81.20% on 10 observed labels.
Validation accuracy: 0.804; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.812.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **89.5%** (0: 94%, 1: 85%) | **89.5%** (0: 94%, 1: 85%) | **89.5%** (0: 94%, 1: 85%) | **89.5%** (0: 94%, 1: 85%) | **89.5%** (0: 94%, 1: 85%) |
| Client 1 | **88.5%** (2: 78%, 3: 99%) | **88.5%** (2: 78%, 3: 99%) | **88.5%** (2: 78%, 3: 99%) | **88.5%** (2: 78%, 3: 99%) | **88.5%** (2: 78%, 3: 99%) |
| Client 2 | **83.0%** (4: 91%, 5: 75%) | **83.0%** (4: 91%, 5: 75%) | **83.0%** (4: 91%, 5: 75%) | **83.0%** (4: 91%, 5: 75%) | **83.0%** (4: 91%, 5: 75%) |
| Client 3 | **72.0%** (6: 67%, 7: 77%) | **72.0%** (6: 67%, 7: 77%) | **72.0%** (6: 67%, 7: 77%) | **72.0%** (6: 67%, 7: 77%) | **72.0%** (6: 67%, 7: 77%) |
| Client 4 | **73.0%** (8: 93%, 9: 53%) | **73.0%** (8: 93%, 9: 53%) | **73.0%** (8: 93%, 9: 53%) | **73.0%** (8: 93%, 9: 53%) | **73.0%** (8: 93%, 9: 53%) |
