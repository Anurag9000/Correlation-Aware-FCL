# Server sync 1 — global model v1

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 25.00% on 4 observed labels.
Validation accuracy: 0.285; seen labels never predicted: 1.
Server-accepted classes: 4; pending private classes: 0.
Accepted-only TEST accuracy: 0.25.
Terminal flush: False. Selected clients: [0, 1].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 |
|:--|:--:|
| Client 0 | **48.0%** (0: 13%, 1: 83%) |
| Client 1 | **2.0%** (2: 0%, 3: 4%) |
