#!/usr/bin/env python3
"""Apply the v8 V100-only operational checkpoint cadence override."""
from __future__ import annotations
import difflib
import hashlib
from pathlib import Path
import sys

OLD = """            if cfg.checkpoint_every and (round_id+1)%cfg.checkpoint_every==0 or \\
               (cfg.stream_protocol or (round_id+1)%cfg.rounds_per_stage==0):
                save_round_state(latest_checkpoint,cfg,round_id,global_state,version,
                                 clients,observed,peak,integrated,history,rng,
                                 server_optimizer_state,server_class_update_counts,server_cl_state)
                if cfg.checkpoint_every:
                    save_round_state(run_dir/f'checkpoint_round_{round_id+1:04d}.pt',
                                     cfg,round_id,global_state,version,
                                     clients,observed,peak,integrated,history,rng,
                                     server_optimizer_state,server_class_update_counts,server_cl_state)
            if (cfg.stream_protocol or (round_id+1)%cfg.rounds_per_stage==0):
                save_checkpoint(run_dir/f'model_stage_{current_stage+1:02d}.pt',global_state,
                                cfg,version,clients,full_state=True,server_cl_state=server_cl_state)
"""
NEW = """            # Operational checkpoint cadence only; all training and evaluation
            # updates still run at every documented logical round.
            checkpoint_interval = max(1, int(os.environ.get('CAFCL_CHECKPOINT_INTERVAL', '1')))
            checkpoint_due = ((round_id + 1) % checkpoint_interval == 0
                              or round_id + 1 == total_rounds)
            if ((cfg.checkpoint_every and (round_id+1)%cfg.checkpoint_every==0) or
                    (cfg.stream_protocol and checkpoint_due) or
                    (not cfg.stream_protocol and (round_id+1)%cfg.rounds_per_stage==0)):
                save_round_state(latest_checkpoint,cfg,round_id,global_state,version,
                                 clients,observed,peak,integrated,history,rng,
                                 server_optimizer_state,server_class_update_counts,server_cl_state)
                if cfg.checkpoint_every:
                    save_round_state(run_dir/f'checkpoint_round_{round_id+1:04d}.pt',
                                     cfg,round_id,global_state,version,
                                     clients,observed,peak,integrated,history,rng,
                                     server_optimizer_state,server_class_update_counts,server_cl_state)
            if ((cfg.stream_protocol and round_id + 1 == total_rounds) or
                    (not cfg.stream_protocol and (round_id+1)%cfg.rounds_per_stage==0)):
                save_checkpoint(run_dir/f'model_stage_{current_stage+1:02d}.pt',global_state,
                                cfg,version,clients,full_state=True,server_cl_state=server_cl_state)
"""

def main():
    if len(sys.argv) != 2:
        raise SystemExit('usage: patch_v100_checkpoint_cadence.py <source_variants_root>')
    root = Path(sys.argv[1])
    diffs = []
    for setting in ('classil_src', 'taskil_src'):
        path = root / setting / 'src/cafcl/simulator.py'
        before = path.read_text()
        if before.count(OLD) != 1:
            raise SystemExit(f'expected one known checkpoint block in {path}; found {before.count(OLD)}')
        after = before.replace(OLD, NEW, 1)
        diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                          fromfile=f'a/{setting}/src/cafcl/simulator.py',
                                          tofile=f'b/{setting}/src/cafcl/simulator.py'))
        path.write_text(after)
        print(f'{setting} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}')
    diff_path = root / 'checkpoint_cadence_v8.patch'
    diff_path.write_text(''.join(diffs))
    print(f'patch={diff_path} sha256={hashlib.sha256(diff_path.read_bytes()).hexdigest()}')

if __name__ == '__main__':
    main()
