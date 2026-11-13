from fractions import Fraction

def choose(rows,cap):
    eligible=[]
    for r in rows:
        ks=[k for k in r['feasible_Ks'] if k<=cap]
        if not ks:continue
        k=min(ks);key=(k,Fraction(r['H_encoded'],r['L_encoded']),r['source']!='CHILD',-r['support_request_count'],r['candidate_id'])
        eligible.append((key,r))
    if not eligible:return None
    key,row=min(eligible,key=lambda v:v[0]);return dict(row,K=key[0])
