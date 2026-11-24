# Server sync 4 — global model v4

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 27.62% on 21 observed labels.
Validation accuracy: 0.26666666666666666; seen labels never predicted: 2.
Server-accepted classes: 21; pending private classes: 0.
Accepted-only TEST accuracy: 0.2761904761904762.
Terminal flush: False. Selected clients: [0, 1, 2, 3, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 |
|:--|:--:|:--:|:--:|:--:|
| Client 0 | **54.5%** (0: 61%, 1: 48%) | **31.5%** (5: 34%, 6: 29%) | **2.0%** (10: 0%, 11: 4%) | **23.5%** (15: 36%, 16: 11%) |
| Client 1 | **45.0%** (1: 48%, 2: 42%) | **41.0%** (6: 29%, 7: 53%) | **33.0%** (11: 4%, 12: 62%) | **10.5%** (16: 11%, 17: 10%) |
| Client 2 | **31.5%** (2: 42%, 3: 21%) | **55.5%** (7: 53%, 8: 58%) | **42.0%** (12: 62%, 13: 22%) | **31.0%** (17: 10%, 18: 52%) |
| Client 3 | **16.5%** (3: 21%, 4: 12%) | **39.0%** (8: 58%, 9: 20%) | **13.5%** (13: 22%, 14: 5%) | **26.0%** (18: 52%, 19: 0%) |
| Client 4 | **23.0%** (4: 12%, 5: 34%) | **10.0%** (9: 20%, 10: 0%) | **20.5%** (14: 5%, 15: 36%) | **0.0%** (19: 0%, 20: 0%) |
