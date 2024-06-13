import pandas as pd
import matplotlib.pyplot as plt
df_train = pd.read_csv('mitbih_train.csv',header=None)
df_test = pd.read_csv('mitbih_test.csv',header=None)
print(df_train.shape)
print(df_test.shape)
#['N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4]
df_train[187]=df_train[187].astype(int)
equilibre=df_train[187].value_counts()
print(equilibre)
plt.figure(figsize=(20,10))
my_circle=plt.Circle( (0,0), 0.7, color='white')
plt.pie(equilibre, labels=['N','Q','V','S','F'], colors=['red','green','blue','skyblue','orange'],autopct='%1.1f%%')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
c=df_train.groupby(187,group_keys=False).apply(lambda train_df : train_df.sample(1))
labb=['Normal (N)','Supra-ventricular premature (S)','Ventricular escape (V)',
      'Fusion of ventricular and normal (F)','Unclassifiable (Q)']
for i in range(5):
    plt.figure()
    plt.plot(c.iloc[i,:186])
    plt.title(labb[i])

