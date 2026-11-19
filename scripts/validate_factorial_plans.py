"""One-pass verification of the source-generated CIFAR-100-only registry."""
from pathlib import Path
import json
from cafcl.full_matrix import manifest


def main():
    expected=json.loads((Path(__file__).resolve().parents[1]/
                         'configs/designs/stream_v4_deit.json').read_text())
    actual=manifest()
    assert actual==expected, 'Active stream design differs from committed manifest'
    assert actual['dataset']=='cifar100'
    assert actual['batch_size']==512
    assert actual['method_aggregator_pairs']==99
    assert actual['job_count']==53805
    print(f"CIFAR-100: {actual['job_count']} source jobs, 99/99 pairs, batch 512; sha256 {actual['sha256']}")


if __name__=='__main__':main()
