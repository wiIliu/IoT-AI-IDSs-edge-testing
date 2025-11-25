import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler

data = pd.read_csv("data/RT_IOT2022.csv")
y = data.pop('Attack_type')

train, X_test, target, y_test = train_test_split(
    data, y, test_size=0.2, random_state=42, stratify=y
)

 
oe = OrdinalEncoder().fit(train[['service']])
train['service'] = oe.transform(train[['service']]).astype(int)
train = pd.get_dummies(train)

 
train['f_Header_b_payload_Ratio'] = train['fwd_header_size_tot'] / train['bwd_pkts_payload.tot'].replace(0, np.nan)
# train['f_Header_b_payload_Ratio'].replace(np.nan, train['f_Header_b_payload_Ratio'].max() + 1) FOR CNN
train['b_Header_f_payload_Ratio'] = train['bwd_header_size_tot'] / train['fwd_pkts_payload.tot'].replace(0, np.nan)
# train['b_Header_f_payload_Ratio'].replace(np.nan, train['b_Header_f_payload_Ratio'].max() + 1) FOR CNN

train["bwd_payload_zero_flg"] = (train["bwd_pkts_payload.tot"] == 0).astype(int)
train["fwd_payload_zero_flg"] = (train["fwd_pkts_payload.tot"] == 0).astype(int)

 
# redundant features or useless
train = train.drop(['proto_udp','fwd_header_size_tot','bwd_pkts_payload.tot',
                    'bwd_header_size_tot','fwd_pkts_payload.tot',"Unnamed: 0",'bwd_URG_flag_count'
                    ],axis=1)

# normalize
scaler = MinMaxScaler().fit(train)
train = pd.DataFrame(scaler.transform(train), columns=train.columns, index=train.index)



# performing preprocessing for test. 
X_test['service'] = oe.transform(X_test[['service']]).astype(int)
X_test = pd.get_dummies(X_test)
X_test['f_Header_b_payload_Ratio'] = X_test['fwd_header_size_tot'] / X_test['bwd_pkts_payload.tot'].replace(0, np.nan)
# train['f_Header_b_payload_Ratio'].replace(np.nan, train['f_Header_b_payload_Ratio'].max() + 1) FOR CNN
X_test['b_Header_f_payload_Ratio'] = X_test['bwd_header_size_tot'] / X_test['fwd_pkts_payload.tot'].replace(0, np.nan)
# train['b_Header_f_payload_Ratio'].replace(np.nan, train['b_Header_f_payload_Ratio'].max() + 1) FOR CNN
X_test["bwd_payload_zero_flg"] = (X_test["bwd_pkts_payload.tot"] == 0).astype(int)
X_test["fwd_payload_zero_flg"] = (X_test["fwd_pkts_payload.tot"] == 0).astype(int)
X_test = X_test.drop(['proto_udp','fwd_header_size_tot','bwd_pkts_payload.tot',
                    'bwd_header_size_tot','fwd_pkts_payload.tot',"Unnamed: 0",'bwd_URG_flag_count'
                    ],axis=1)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)


# export
target = target.loc[train.index]
y_test = y_test.loc[X_test.index]
os.makedirs("data", exist_ok=True) 
pd.concat([train, target], axis=1).to_csv("data/preprocessedTrain1.csv", index=False)
pd.concat([X_test, y_test], axis=1).to_csv("data/preprocessedTest1.csv", index=False)
print("successful")
