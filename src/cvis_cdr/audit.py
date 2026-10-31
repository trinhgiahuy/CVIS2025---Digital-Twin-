"""Compute reviewer-relevant counts from VERIFIED provenance metadata.

Counts frames, unique session IDs, unique participant IDs, and independent
recording files. Never infers that adjacent frames or same-session multiple
captures are independent measurements. 'Simultaneous' requires acquisition
metadata and cannot be inferred from multiple persons in radar FOV.
"""
import argparse,json
from pathlib import Path
from .records import read_manifest,validate_manifest


def counts_for(df):
    grouped=[]
    for (domain,split,label),g in df.groupby(['domain','split','label']):
        sess=set(str(x).strip() for x in g.session_id if str(x).strip())
        part=set(str(x).strip() for x in g.get('participant_id',[]) if str(x).strip())
        grouped.append({'domain':str(domain),'split':str(split),'class_label':int(label),
                        'frames_or_rows':len(g),'sessions_with_known_ids':len(sess),
                        'participant_ids_with_known_values':len(part),
                        'all_sessions_identified':bool((g.session_id!='').all()),
                        'all_participants_identified':bool('participant_id' in g and (g.participant_id!='').all()),
                        'session_ids':sorted(sess)})
    return grouped

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',default='audit.json');a=p.parse_args()
    df=read_manifest(a.manifest)
    validity=validate_manifest(df)
    rows=counts_for(df)
    out={'source_manifest':a.manifest,'rows':rows,'warnings':validity['warnings'],
         'simultaneous_recordings':'UNKNOWN unless verified against original acquisition logs',
         'independent_measurements':'UNKNOWN unless original session/run identifiers and procedure are verified',
         'fig8_paper_binary_test_expected_frames':{'0':427,'1':184},
         'fig9_paper_counting_test_expected_frames':{'0':427,'1':184,'2':153}}
    Path(a.out).write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
