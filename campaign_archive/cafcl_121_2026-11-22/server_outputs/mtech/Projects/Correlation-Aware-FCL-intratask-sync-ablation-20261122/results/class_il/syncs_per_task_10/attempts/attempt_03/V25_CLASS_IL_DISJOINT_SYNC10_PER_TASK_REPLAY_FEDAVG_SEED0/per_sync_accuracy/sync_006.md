# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 45.20% on 10 observed labels.
Validation accuracy: 0.402; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.452.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **58.0%** (0: 63%, 1: 53%) | **58.0%** (0: 63%, 1: 53%) | **58.0%** (0: 63%, 1: 53%) | **58.0%** (0: 63%, 1: 53%) | **58.0%** (0: 63%, 1: 53%) | **58.0%** (0: 63%, 1: 53%) |
| Client 1 | **38.0%** (2: 54%, 3: 22%) | **38.0%** (2: 54%, 3: 22%) | **38.0%** (2: 54%, 3: 22%) | **38.0%** (2: 54%, 3: 22%) | **38.0%** (2: 54%, 3: 22%) | **38.0%** (2: 54%, 3: 22%) |
| Client 2 | **52.5%** (4: 72%, 5: 33%) | **52.5%** (4: 72%, 5: 33%) | **52.5%** (4: 72%, 5: 33%) | **52.5%** (4: 72%, 5: 33%) | **52.5%** (4: 72%, 5: 33%) | **52.5%** (4: 72%, 5: 33%) |
| Client 3 | **31.0%** (6: 40%, 7: 22%) | **31.0%** (6: 40%, 7: 22%) | **31.0%** (6: 40%, 7: 22%) | **31.0%** (6: 40%, 7: 22%) | **31.0%** (6: 40%, 7: 22%) | **31.0%** (6: 40%, 7: 22%) |
| Client 4 | **46.5%** (8: 69%, 9: 24%) | **46.5%** (8: 69%, 9: 24%) | **46.5%** (8: 69%, 9: 24%) | **46.5%** (8: 69%, 9: 24%) | **46.5%** (8: 69%, 9: 24%) | **46.5%** (8: 69%, 9: 24%) |
