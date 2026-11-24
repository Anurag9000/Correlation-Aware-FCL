# Server sync 6 — global model v6

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 48.20% on 10 observed labels.
Validation accuracy: 0.426; seen labels never predicted: 0.
Server-accepted classes: 10; pending private classes: 0.
Accepted-only TEST accuracy: 0.482.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **67.0%** (0: 77%, 1: 57%) | **67.0%** (0: 77%, 1: 57%) | **67.0%** (0: 77%, 1: 57%) | **67.0%** (0: 77%, 1: 57%) | **67.0%** (0: 77%, 1: 57%) | **67.0%** (0: 77%, 1: 57%) |
| Client 1 | **37.5%** (2: 48%, 3: 27%) | **37.5%** (2: 48%, 3: 27%) | **37.5%** (2: 48%, 3: 27%) | **37.5%** (2: 48%, 3: 27%) | **37.5%** (2: 48%, 3: 27%) | **37.5%** (2: 48%, 3: 27%) |
| Client 2 | **50.0%** (4: 72%, 5: 28%) | **50.0%** (4: 72%, 5: 28%) | **50.0%** (4: 72%, 5: 28%) | **50.0%** (4: 72%, 5: 28%) | **50.0%** (4: 72%, 5: 28%) | **50.0%** (4: 72%, 5: 28%) |
| Client 3 | **39.0%** (6: 47%, 7: 31%) | **39.0%** (6: 47%, 7: 31%) | **39.0%** (6: 47%, 7: 31%) | **39.0%** (6: 47%, 7: 31%) | **39.0%** (6: 47%, 7: 31%) | **39.0%** (6: 47%, 7: 31%) |
| Client 4 | **47.5%** (8: 65%, 9: 30%) | **47.5%** (8: 65%, 9: 30%) | **47.5%** (8: 65%, 9: 30%) | **47.5%** (8: 65%, 9: 30%) | **47.5%** (8: 65%, 9: 30%) | **47.5%** (8: 65%, 9: 30%) |
