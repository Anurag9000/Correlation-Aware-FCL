# Server sync 2 — global model v2

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 42.94% on 18 observed labels.
Validation accuracy: 0.44555555555555554; seen labels never predicted: 1.
Server-accepted classes: 18; pending private classes: 0.
Accepted-only TEST accuracy: 0.42944444444444446.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 |
|:--|:--:|:--:|
| Client 0 | **36.4%** (4: 5%, 7: 49%, 15: 22%, 58: 15%, 63: 91%) | **29.2%** (12: 10%, 18: 70%, 71: 8%, 86: 0%, 96: 58%) |
| Client 1 | **27.4%** (4: 5%, 7: 49%, 40: 2%, 58: 15%, 95: 66%) | **36.8%** (12: 10%, 16: 46%, 18: 70%, 86: 0%, 96: 58%) |
| Client 2 | **18.6%** (4: 5%, 7: 49%, 15: 22%, 40: 2%, 58: 15%) | **23.6%** (12: 10%, 55: 100%, 71: 8%, 86: 0%, 91: 0%) |
| Client 3 | **28.4%** (4: 5%, 40: 2%, 58: 15%, 69: 54%, 95: 66%) | **40.8%** (12: 10%, 16: 46%, 18: 70%, 26: 78%, 86: 0%) |
| Client 4 | **45.2%** (4: 5%, 40: 2%, 69: 54%, 80: 99%, 95: 66%) | **17.6%** (12: 10%, 18: 70%, 71: 8%, 86: 0%, 91: 0%) |
