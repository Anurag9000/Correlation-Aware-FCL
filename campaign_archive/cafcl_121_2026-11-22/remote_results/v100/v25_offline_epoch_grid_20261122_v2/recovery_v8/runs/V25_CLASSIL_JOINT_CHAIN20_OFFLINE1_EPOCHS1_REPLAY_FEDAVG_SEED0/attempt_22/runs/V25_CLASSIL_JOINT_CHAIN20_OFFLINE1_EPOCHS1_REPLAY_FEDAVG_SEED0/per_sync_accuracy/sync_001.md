# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 28.83% on 6 observed labels.
Validation accuracy: 0.25666666666666665; seen labels never predicted: 1.
Server-accepted classes: 5; pending private classes: 1.
Accepted-only TEST accuracy: 0.346.
Terminal flush: False. Selected clients: [0, 1, 2, 3].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **82.0%** (0: 76%, 1: 88%) |
| Client 1 | **46.0%** (1: 88%, 2: 4%) |
| Client 2 | **4.5%** (2: 4%, 3: 5%) |
| Client 3 | **2.5%** (3: 5%, 4: 0%) |
| Client 4 | **0.0%** (4: 0%, 5: 0%) [PENDING upload] |
