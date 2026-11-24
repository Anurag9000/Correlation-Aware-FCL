# Server sync 7 — global model v7

Backbone: resnet18_cifar32; held-out CIFAR-100 TEST diagnostic only.
Seen-class global accuracy: 57.31% on 36 observed labels.
Validation accuracy: 0.5755555555555556; seen labels never predicted: 0.
Server-accepted classes: 32; pending private classes: 4.
Accepted-only TEST accuracy: 0.6390625.
Terminal flush: False. Selected clients: [0, 4].
**Test diagnostics MUST NOT control training, optimizer choice, selection, or tuning.**

| Client | U1 | U2 | U3 | U4 | U5 | U6 | U7 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Client 0 | **69.0%** (0: 98%, 1: 40%) | **50.0%** (5: 63%, 6: 37%) | **50.0%** (10: 35%, 11: 65%) | **50.0%** (15: 0%, 16: 100%) | **85.5%** (20: 73%, 21: 98%) | **54.0%** (25: 11%, 26: 97%) | **86.5%** (30: 97%, 31: 76%) |
| Client 1 | **60.5%** (1: 40%, 2: 81%) | **47.0%** (6: 37%, 7: 57%) | **78.0%** (11: 65%, 12: 91%) | **92.5%** (16: 100%, 17: 85%) | **57.5%** (21: 98%, 22: 17%) | **48.5%** (26: 97%, 27: 0%) [PENDING upload] | **38.0%** (31: 76%, 32: 0%) [PENDING upload] |
| Client 2 | **77.0%** (2: 81%, 3: 73%) | **66.5%** (7: 57%, 8: 76%) | **46.5%** (12: 91%, 13: 2%) | **52.0%** (17: 85%, 18: 19%) | **54.0%** (22: 17%, 23: 91%) | **0.5%** (27: 0%, 28: 1%) [PENDING upload] | **39.0%** (32: 0%, 33: 78%) [PENDING upload] |
| Client 3 | **61.5%** (3: 73%, 4: 50%) | **71.0%** (8: 76%, 9: 66%) | **16.5%** (13: 2%, 14: 31%) | **54.5%** (18: 19%, 19: 90%) | **74.5%** (23: 91%, 24: 58%) | **34.0%** (28: 1%, 29: 67%) [PENDING upload] | **78.0%** (33: 78%, 34: 78%) [PENDING upload] |
| Client 4 | **56.5%** (4: 50%, 5: 63%) | **50.5%** (9: 66%, 10: 35%) | **15.5%** (14: 31%, 15: 0%) | **81.5%** (19: 90%, 20: 73%) | **34.5%** (24: 58%, 25: 11%) | **82.0%** (29: 67%, 30: 97%) | **70.0%** (34: 78%, 35: 62%) |
