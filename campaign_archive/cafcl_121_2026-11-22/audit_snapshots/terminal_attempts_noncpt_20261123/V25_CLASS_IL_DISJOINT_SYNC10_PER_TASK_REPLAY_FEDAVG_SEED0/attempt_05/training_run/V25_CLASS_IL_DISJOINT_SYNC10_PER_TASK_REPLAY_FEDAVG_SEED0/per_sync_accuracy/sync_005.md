# Server sync 5 — global model v5

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 44.10% on 10 observed labels.
Validation accuracy: 0.378; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.441.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 |
|:--|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **61.0%** (0: 61%, 1: 61%) | **61.0%** (0: 61%, 1: 61%) | **61.0%** (0: 61%, 1: 61%) | **61.0%** (0: 61%, 1: 61%) | **61.0%** (0: 61%, 1: 61%) |
| Client 1 | **46.0%** (2: 42%, 3: 50%) | **46.0%** (2: 42%, 3: 50%) | **46.0%** (2: 42%, 3: 50%) | **46.0%** (2: 42%, 3: 50%) | **46.0%** (2: 42%, 3: 50%) |
| Client 2 | **28.0%** (4: 45%, 5: 11%) | **28.0%** (4: 45%, 5: 11%) | **28.0%** (4: 45%, 5: 11%) | **28.0%** (4: 45%, 5: 11%) | **28.0%** (4: 45%, 5: 11%) |
| Client 3 | **32.0%** (6: 32%, 7: 32%) | **32.0%** (6: 32%, 7: 32%) | **32.0%** (6: 32%, 7: 32%) | **32.0%** (6: 32%, 7: 32%) | **32.0%** (6: 32%, 7: 32%) |
| Client 4 | **53.5%** (8: 56%, 9: 51%) | **53.5%** (8: 56%, 9: 51%) | **53.5%** (8: 56%, 9: 51%) | **53.5%** (8: 56%, 9: 51%) | **53.5%** (8: 56%, 9: 51%) |
