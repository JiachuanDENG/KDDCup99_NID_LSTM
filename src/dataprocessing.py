import pandas as pd
import numpy as np
from time import time
from targetEncoder import TargetEncoder
from sklearn.preprocessing import StandardScaler
import configparser

print ('loading data...')
config=configparser.ConfigParser()
config.read('./config.ini')

col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
DoS = ['back.','land.','neptune.','pod.','smurf.', 'teardrop.']
R2L = ['ftp_write.','guess_passwd.','imap.','multihop.','phf.','spy.','warezclient.','warezmaster.']
U2R = ['buffer-overflow.','loadmodule.','perl.','rootkit.']
Probe = ['ipsweep.','nmap.','portsweep.','satan.']

kdd_data= pd.read_csv(config.get('DataPath','traindatapath'), header=None, names = col_names)

kdd_data.loc[kdd_data['label']!='normal.','label']=0
kdd_data.loc[kdd_data['label']=='normal.','label']=1

print ('normal:',len(kdd_data[kdd_data['label']==1]),'abnormal:',len(kdd_data[kdd_data['label']!=1]))
# according to https://www.researchgate.net/profile/Ralf_Staudemeyer/publication/279770740_Applying_long_short-term_memory_recurrent_neural_networks_to_intrusion_detection/links/559a972908ae99aa62ce1468/Applying-long-short-term-memory-recurrent-neural-networks-to-intrusion-detection.pdf
#   only few columns are useful to construct a time-series-like data
useful_columns=['service', 'src_bytes', 'dst_host_diff_srv_rate',\
'dst_host_rerror_rate', 'dst_bytes', 'hot', 'num_failed_logins', 'dst_host_srv_count','label']

kdd_data=kdd_data[useful_columns]

print('shuffle train df...')
kdd_data = kdd_data.reindex(np.random.permutation(kdd_data.index)).reset_index(drop=True)

tr_ratio=float(config.get('Parameters','train_ratio'))
tr_len=int(len(kdd_data)*(tr_ratio))
print ('training len:{},val len:{}'.format(tr_len,len(kdd_data)-tr_len))

print ('encoding categorical features...')
target_encoder=TargetEncoder(kdd_data[:tr_len],kdd_data[tr_len:],10,10,0.01)
service_encode=target_encoder.encode1col('service')
kdd_data['service']=service_encode

features=['service','src_bytes','dst_host_diff_srv_rate','dst_host_rerror_rate','dst_bytes','hot','num_failed_logins','dst_host_srv_count']

X=np.array(kdd_data[features])

y=np.array(kdd_data['label']).reshape([-1,1])
print ('label to onehot')
# tensorflow only accept one-hot-encode labels
y_one_hot=np.zeros([y.shape[0],2])
for i in range(y.shape[0]):
	if y[i,0]==1:
		y_one_hot[i,1]=1
	else:
		y_one_hot[i,0]=1

scaler = StandardScaler()
scaler.fit(X)
X=scaler.transform(X)
X=X.reshape([-1,X.shape[1],1])
X_tr,y_tr=X[:tr_len,:],y_one_hot[:tr_len,:]
X_val,y_val=X[tr_len:,:],y_one_hot[tr_len:,:]

print ('save data...')
np.save(config.get('DataPath','X_tr_path'),X_tr)
np.save(config.get('DataPath','y_tr_path'),y_tr)
np.save(config.get('DataPath','X_val_path'),X_val)
np.save(config.get('DataPath','y_val_path'),y_val)








