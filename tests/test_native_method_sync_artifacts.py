"""Source-specific GLFC/LGA/FEAT method history synchronized only at upload."""
from types import SimpleNamespace
import torch
from cafcl.continual import LocalState
from cafcl.federated_cl import FederatedCLState


class Bank:
    class Train:
        targets=[0,0,1,1,2,2,3,3]
        def __len__(self):return len(self.targets)
        def __getitem__(self,i):return torch.zeros(3,32,32),self.targets[i]
    train=Train()


def test_offline_native_entropy_and_feat_tickets_are_not_published_early():
    first=LocalState(0)
    hidden=LocalState(1)
    first.last_learning_method='native_glfc'
    hidden.last_learning_method='native_glfc'
    first.native_memory_state={'last_entropy':1.,'learned_classes':[0,1]}
    hidden.native_memory_state={'last_entropy':3.,'learned_classes':[2,3]}
    first.pending_upload_classes={0,1}
    hidden.pending_upload_classes={2,3}
    server=FederatedCLState()
    server.merge_clients([first],Bank(),'native_glfc',
                         {'w':torch.tensor([1.])},1)
    assert set(server.source_method_artifacts)=={0}
    assert hidden.native_memory_state['last_entropy']==3.
    first.receive({'w':torch.tensor([1.])},1)
    server.broadcast_to(first,'native_glfc')
    assert set(first.received_native_artifacts)=={0}
    assert 1 not in server.source_method_artifacts
    server.assert_complete_download(first)
    server.merge_clients([hidden],Bank(),'native_glfc',
                         {'w':torch.tensor([2.])},2)
    assert set(server.source_method_artifacts)=={0,1}
    hidden.receive({'w':torch.tensor([2.])},2)
    server.broadcast_to(hidden,'native_glfc')
    assert hidden.received_native_artifacts[1]['native_memory_state'][
        'last_entropy']==3.
    server.assert_complete_download(hidden)
    assert hidden.native_memory_state['last_entropy']==3.


def test_native_source_receipts_are_deep_copied_and_not_aliased():
    feat=LocalState(3)
    feat.last_learning_method='native_feat'
    feat.feat_selection_tickets={4:[12,12,13]}
    feat.feat_selection_receipts=[{'classes':[4],'selection':{4:{'tickets':3}}}]
    server=FederatedCLState()
    server.merge_clients([feat],Bank(),'native_feat',
                         {'w':torch.tensor([0.])},1)
    stored=server.source_method_artifacts[3]
    feat.feat_selection_tickets[4].append(99)
    assert stored['feat_selection_tickets'][4]==[12,12,13]
    feat.receive({'w':torch.tensor([0.])},1)
    server.broadcast_to(feat,'native_feat')
    feat.received_native_artifacts[3]['feat_selection_tickets'][4].clear()
    assert stored['feat_selection_tickets'][4]==[12,12,13]
