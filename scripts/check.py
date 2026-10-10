python3 -c "
import json
data = json.load(open('/mnt/yulan_pretrain_131/clinical_project/radclaw/output/ablation_retrieval_only/predictions_vs_gt.json'))
print(len(data[0]['prediction']), 'chars')
print(data[0]['prediction'][:500])
"
python3 -c "
import json
data = json.load(open('/mnt/yulan_pretrain_131/clinical_project/radclaw/output/ablation_retrieval_only/predictions_vs_gt.json'))
print('样本数:', len(data))
"
python3 - << 'EOF'
from transformers import BertTokenizer
import json

tok = BertTokenizer.from_pretrained("bert-base-uncased")
data = json.load(open("/mnt/yulan_pretrain_131/clinical_project/radclaw/output/ablation_retrieval_only/predictions_vs_gt.json"))

for i, d in enumerate(data):
    pred_len = len(tok.encode(d["prediction"]))
    ref_len  = len(tok.encode(d["ground_truth"]))
    if pred_len > 400 or ref_len > 400:
        print(f"[{i}] pred={pred_len}, ref={ref_len}")
        print(f"     pred前300字: {d['prediction'][:300]}")
        print()
EOF