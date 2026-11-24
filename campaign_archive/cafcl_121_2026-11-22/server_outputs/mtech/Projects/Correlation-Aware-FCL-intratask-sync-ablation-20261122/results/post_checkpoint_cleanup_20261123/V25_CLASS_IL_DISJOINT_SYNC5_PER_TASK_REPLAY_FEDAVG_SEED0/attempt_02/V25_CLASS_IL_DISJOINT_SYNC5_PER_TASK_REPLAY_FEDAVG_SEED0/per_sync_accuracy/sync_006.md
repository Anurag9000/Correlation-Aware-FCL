# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 27.75% on 20 observed labels.
Validation accuracy: 0.256; seen labels never predicted: 6.
Server-accepted classes: 20; pending private classes: 0.
Accepted-only TEST accuracy: 0.2775.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **69.0%** (0: 80%, 1: 58%) | **69.0%** (0: 80%, 1: 58%) | **69.0%** (0: 80%, 1: 58%) | **69.0%** (0: 80%, 1: 58%) | **69.0%** (0: 80%, 1: 58%) | **1.0%** (10: 0%, 11: 2%) |
| Client 1 | **55.0%** (2: 62%, 3: 48%) | **55.0%** (2: 62%, 3: 48%) | **55.0%** (2: 62%, 3: 48%) | **55.0%** (2: 62%, 3: 48%) | **55.0%** (2: 62%, 3: 48%) | **0.5%** (12: 0%, 13: 1%) |
| Client 2 | **46.5%** (4: 44%, 5: 49%) | **46.5%** (4: 44%, 5: 49%) | **46.5%** (4: 44%, 5: 49%) | **46.5%** (4: 44%, 5: 49%) | **46.5%** (4: 44%, 5: 49%) | **1.5%** (14: 3%, 15: 0%) |
| Client 3 | **48.5%** (6: 50%, 7: 47%) | **48.5%** (6: 50%, 7: 47%) | **48.5%** (6: 50%, 7: 47%) | **48.5%** (6: 50%, 7: 47%) | **48.5%** (6: 50%, 7: 47%) | **0.5%** (16: 0%, 17: 1%) |
| Client 4 | **55.0%** (8: 69%, 9: 41%) | **55.0%** (8: 69%, 9: 41%) | **55.0%** (8: 69%, 9: 41%) | **55.0%** (8: 69%, 9: 41%) | **55.0%** (8: 69%, 9: 41%) | **0.0%** (18: 0%, 19: 0%) |
