"""Central read-only analysis across validated campaign attempts."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from .diagnostic_campaign import atomic_json,effective_jobs,signature,validate_artifacts


def mean(xs):
    xs=[x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def latest_valid(job,root):
    for p in sorted((root/job["family"]/job["job_id"]).glob("attempt_*/status.json"),reverse=True):
        try:
            s=json.loads(p.read_text())
            if s["signature"]!=signature(job) or s["state"]!="completed":continue
            validate_artifacts(job,p.parent)
            return p.parent
        except (OSError,ValueError,KeyError,TypeError):continue
    return None


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    keys=sorted({k for r in rows for k in r}) or ["job_id"]
    temp=path.with_suffix(".tmp")
    with temp.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    temp.replace(path)


def acquisition_conditioned(path):
    ev=[json.loads(p.read_text())["test"] for p in sorted(path.glob("stage_*_evaluation.json"))]
    if not ev:return {}
    final_path=path/"terminal_evaluation.json"
    final=json.loads(final_path.read_text()) if final_path.is_file() else ev[-1]
    peaks={k:max(x[k]["accuracy"] for x in ev if k in x) for k in final if any(k in x for x in ev)}
    out={}
    for threshold in (.5,.7,.9):
        keys=[k for k,p in peaks.items() if p>=threshold]
        suffix=str(int(threshold*100))
        out["qualified_"+suffix]=len(keys)
        out["forgetting_at_"+suffix]=mean([max(0.,peaks[k]-final[k]["accuracy"]) for k in keys])
    if len(ev)>1:
        mid=ev[(len(ev)-1)//2]
        keys=set(mid)&set(final)
        out["fixed_mid_history_accuracy"]=mean([mid[k]["accuracy"] for k in keys])
        out["fixed_mid_final_accuracy"]=mean([final[k]["accuracy"] for k in keys])
    return out


def collect(root,jobs):
    summaries=[];trajectory=[];unavailable=[]
    for j in jobs:
        if not j.get("binding"):continue
        p=latest_valid(j,root)
        if p is None:
            unavailable.append(j["job_id"]);continue
        s=json.loads((p/"result_summary.json").read_text())
        row={k:j.get(k) for k in ("job_id","family","clients","il","partition","objective","curriculum","participation",
                                  "rounds_per_stage","local_epochs","schedule","intervention","binding")}
        row["attempt"]=str(p.relative_to(root))
        if j["binding"]=="stationary_reference":
            row.update(final_accuracy=s["final"]["test_accuracy"],final_loss=s["final"]["test_loss"],
                optimizer_steps=s["final"]["optimizer_steps"],processed_examples=s["final"]["processed_examples"])
            with (p/"metrics.csv").open(newline="") as f:
                trajectory.extend({"job_id":j["job_id"],"binding":j["binding"],**x} for x in csv.DictReader(f))
        else:
            row.update(final_accuracy=s["final_accuracy"],current_task_average_accuracy=s["current_task_average_accuracy"],
                final_forgetting=s["final_forgetting"],legacy_union_accuracy=s.get("legacy_union_accuracy"),**s["counts"])
            diags=[json.loads(x.read_text()) for x in sorted((p/"per_round_metrics").glob("sync_*.json"))]
            row["mean_aggregation_forgetting"]=mean([x.get("aggregation_forgetting") for x in diags])
            row["mean_positive_aggregation_forgetting"]=mean([max(0.,x["aggregation_forgetting"]) for x in diags if x.get("aggregation_forgetting") is not None])
            row["mean_local_forgetting"]=mean([v.get("local_forgetting") for x in diags for v in x.get("local",[])])
            row["mean_download_forgetting"]=mean([v.get("download_history_drop") for x in diags for v in x.get("download",[])])
            row["mean_upload_lag"]=mean([v["lag"] for x in diags for v in x.get("uploads",[])])
            row["maximum_pending_classes"]=max((len(x.get("pending_classes",[])) for x in diags),default=0)
            row.update(acquisition_conditioned(p))
            for x in diags:
                tr={"job_id":j["job_id"],"binding":j["binding"],**{k:x.get(k) for k in
                    ("sync","stage","round_in_stage","flush","history_before","history_after","aggregation_forgetting","current_before","current_after",
                     "test_current","test_history","test_global","test_forgetting","legacy_union_accuracy","current_loss_before","current_loss_after")}}
                for key in ("gradient","post_current_gradient","pre_current_replay_gradient","post_current_replay_gradient"):
                    g=x.get(key) or {}
                    tr[key+"_norm_squared"]=g.get("sampled_weighted_current_objective_gradient_norm_squared")
                    tr[key+"_loss"]=g.get("sampled_weighted_objective_loss")
                trajectory.append(tr)
        summaries.append(row)
    return summaries,trajectory,unavailable


def contrasts(rows):
    out=[]
    def compare(name,a,b):
        if a is None or b is None:return
        rec={"contrast":name,"a":a["job_id"],"b":b["job_id"]}
        for metric in ("final_accuracy","current_task_average_accuracy","final_forgetting","legacy_union_accuracy","mean_aggregation_forgetting"):
            if a.get(metric) is not None and b.get(metric) is not None:rec[metric+"_a_minus_b"]=a[metric]-b[metric]
        out.append(rec)
    def pick(**kw):
        found=[x for x in rows if all(x.get(k)==v for k,v in kw.items())]
        return found[0] if len(found)==1 else None
    for k in (2,3,10):
        compare("stationary global minus restricted",pick(family="FL",clients=k,partition="disjoint_labels",objective="global"),pick(family="FL",clients=k,partition="disjoint_labels",objective="restricted_local"))
        for r in (1,5,10):
            compare("CC global seen minus original",pick(family="CC",clients=k,il="class_il",rounds_per_stage=r,objective="global_seen"),pick(family="CC",clients=k,il="class_il",rounds_per_stage=r,objective="original"))
    for il in ("class_il","task_il"):
        for participation in (.4,.6):
            compare("JR joint minus disjoint",pick(family="JR",curriculum="joint",il=il,participation=participation),pick(family="JR",curriculum="disjoint",il=il,participation=participation))
            for curriculum in ("matched_partial_class_overlap","matched_complete_task_overlap"):
                compare("MO overlap minus disjoint",pick(family="MO",curriculum=curriculum,il=il,participation=participation),pick(family="MO",curriculum="matched_disjoint",il=il,participation=participation))
            for curriculum in ("joint","disjoint"):
                for r in (5,10):
                    compare("JC more communication minus R1",pick(family="JC",curriculum=curriculum,il=il,participation=participation,rounds_per_stage=r),pick(family="JR",curriculum=curriculum,il=il,participation=participation))
        for a,b in (("high_fresh","none_fresh"),("high_stale","none_stale"),("high_fresh","high_stale"),("none_fresh","none_stale")):
            compare("OF "+a+" minus "+b,pick(family="OF",schedule=a,il=il),pick(family="OF",schedule=b,il=il))
        for x in rows:
            if x["family"]=="ST" and x["il"]==il:
                compare("ST "+x["intervention"]+" minus reconstructed baseline",x,pick(family="JR",curriculum="joint",il=il,participation=.4))
    return out


def compute_matching(rows):
    groups=defaultdict(list)
    for r in rows:
        if r["family"]=="CM":groups[(r["clients"],r["il"],r["partition"])].append(r)
    out=[]
    for key,rs in groups.items():
        fields=("optimizer_steps","unique_current_visits","current_draws","replay_draws")
        out.append({"clients":key[0],"il":key[1],"partition":key[2],"jobs":[r["job_id"] for r in rs],
            "complete":len(rs)==3,"counts_match":len(rs)==3 and all(len({r.get(f) for r in rs})==1 for f in fields),
            "counts":{r["job_id"]:{f:r.get(f) for f in fields} for r in rs}})
    return out


def analyze(root,jobs):
    root=Path(root);out=root/"analysis";out.mkdir(parents=True,exist_ok=True)
    summaries,trajectory,unavailable=collect(root,jobs)
    comparison=contrasts(summaries);budget=compute_matching(summaries)
    write_csv(out/"job_summary.csv",summaries)
    write_csv(out/"learning_trajectories.csv",trajectory)
    write_csv(out/"controlled_contrasts.csv",comparison)
    atomic_json(out/"compute_matching.json",budget)
    result={"completed_jobs":len(summaries),"unavailable_jobs":unavailable,
            "controlled_contrasts":len(comparison),"compute_matching":budget,
            "note":"Descriptive single-seed results. Historical reconstruction is not exact reproduction; all contrasts use the same reference binding."}
    atomic_json(out/"analysis_summary.json",result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",default="runs/diagnostic_campaign")
    p.add_argument("--allow-reconstruction",action="store_true");p.add_argument("--subset-probes",action="store_true")
    a=p.parse_args()
    print(json.dumps(analyze(Path(a.output).resolve(),effective_jobs(a.allow_reconstruction,a.subset_probes)),indent=2))


if __name__=="__main__":main()
