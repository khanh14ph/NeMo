from nemo.collections.asr.models import ASRModel
from omegaconf import OmegaConf
decoding_config=OmegaConf.load("/home4/khanhnd/nemo_dev/config/decode_config.yaml")
choose="rnnt"
checkpoint1="/home4/khanhnd/lightning_logs/nemo/FastConformer-Hybrid-TDT-CTC-BPE/2025-05-20_12-28-24/checkpoints/FastConformer-Hybrid-TDT-CTC-BPE.nemo"
checkpoint2="/home4/khanhnd/lightning_logs/nemo/FastConformer-Hybrid-TDT-CTC-BPE-mix/2025-05-20_22-19-02/checkpoints/FastConformer-Hybrid-TDT-CTC-BPE-mix.nemo"
asr_model1 = ASRModel.restore_from(checkpoint1)
asr_model2 = ASRModel.restore_from(checkpoint2)
import pandas as pd
from tqdm import tqdm
res1=[]
res2=[]
df=pd.read_csv("/home4/khanhnd/hieupt/thesis/data/csv/fleurs_dev.csv")
for idx,i in tqdm(df.iterrows()):

# asr_model.change_decoding_strategy(decoding_config[choose],choose)
    output1 = asr_model1.transcribe([i["path"]])[0].text
    output2= asr_model2.transcribe([i["path"]])[0].text
    res1.append(output1)
    res2.append(output2)
df["pred_concat"]=res1
df["pred_agg"]=res2
df.to_csv("temp.tsv",sep="\t")