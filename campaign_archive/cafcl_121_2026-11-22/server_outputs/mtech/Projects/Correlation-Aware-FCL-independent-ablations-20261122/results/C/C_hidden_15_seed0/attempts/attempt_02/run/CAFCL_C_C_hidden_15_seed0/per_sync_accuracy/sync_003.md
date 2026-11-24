# Server sync 3 — global model v3

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 74.07% on 30 observed labels.
Validation accuracy: 0.754; seen labels never predicted: 2.
Server-accepted classes: 24; pending private classes: 6.
Accepted-only TEST accuracy: 0.8008333333333333.
Terminal flush: False. Selected clients: [0, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 |
|:--|:--:|:--:|:--:|
| Client 0 | **81.5%** (8: 96%, 9: 67%) | **75.5%** (14: 66%, 15: 85%) | **65.0%** (26: 31%, 27: 99%) |
| Client 1 | **50.0%** (6: 0%, 7: 100%) [PENDING upload] | **50.0%** (18: 0%, 19: 100%) [PENDING upload] | **50.0%** (28: 66%, 29: 34%) [PENDING upload] |
| Client 2 | **83.5%** (4: 87%, 5: 80%) | **93.5%** (16: 91%, 17: 96%) | **73.0%** (22: 50%, 23: 96%) |
| Client 3 | **87.5%** (0: 82%, 1: 93%) | **74.0%** (10: 83%, 11: 65%) | **90.5%** (20: 83%, 21: 98%) |
| Client 4 | **85.0%** (2: 76%, 3: 94%) | **71.5%** (12: 76%, 13: 67%) | **80.5%** (24: 61%, 25: 100%) |
