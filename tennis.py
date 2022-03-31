import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import binomtest
import os
import regex as re
import sklearn
import matplotlib.pyplot as plt

print ('start')
#print(os.getcwd())

df =pd.read_csv(r".\input\atp.csv")
print(df.info())
print(df.shape)

#remove columns not in scope
df = df.drop(['loser_ioc', 'winner_ioc', 'draw_size', 'minutes', 'winner_entry','tourney_id', "loser_entry", 'match_num'], axis=1)
df_hist = df
df = df.drop(df[df['tourney_date'] < 20140101].index)
print(df.shape)

#missing values
null_percent = df.isnull().sum() * 100 / len(df)
null_entries = df.isnull().sum()
missing_v_df = pd.DataFrame({'column_name': df.columns,'missing' : null_entries ,'missing %': null_percent})
print(missing_v_df)

df=df.drop(['winner_seed', 'loser_seed', 'winner_ht', 'loser_ht'], axis=1)
df=df.dropna()
print(df.info())
print(df.isnull().sum())

#Feature engineering

print(df['score'].sample(20))
df['set'] = df[['score']].applymap(lambda x: str.count(x,'-'))
print(df['set'].drop_duplicates())
df['retired'] = df[['score']].applymap(lambda x: str.count(x,'RET'))
df = df.drop(df[(df['retired'] == 1) | (df['set'] == 0 )].index)
df = df.drop(['retired'],axis=1)
print(df['set'].drop_duplicates())
print(df.shape)

df['rank_diff'] = df['winner_rank_points'] - df['loser_rank_points']
df['rank_diff'] = df['rank_diff'].abs()
df['underdog_win'] = np.where(df['winner_rank_points'] > df['loser_rank_points'], "no", "yes")

match_stats_list = ['l_1stIn','l_1stWon' ,'l_2ndWon' ,'l_SvGms' ,'l_ace' ,'l_bpFaced' ,'l_bpSaved' ,'l_df'
,'l_svpt' ,'w_1stIn' ,'w_1stWon' ,'w_2ndWon' ,'w_SvGms' ,'w_ace','w_bpFaced','w_bpSaved','w_df','w_svpt']

df[match_stats_list] = df[match_stats_list].apply(lambda x: (x / df['set']).round(2))

#identify and remove highly correlated variables

def correlation_mat (d):
    corrmat = d.corr().round(2)
    f, ax = plt.subplots(figsize=(24, 14))
    colormap = sns.color_palette("coolwarm", as_cmap=True)
    sns.heatmap(corrmat, vmax=1, cmap=colormap, square=True, annot=True,linecolor='black',linewidths = 0.01, annot_kws={'fontsize': 7})
    plt.title('Pearson Correlation of all numerical variables', y=1.05, size=15)
    plt.show()

correlation_mat(df)

df = df.drop(['l_1stIn','l_1stWon'  ,'l_SvGms'  ,'l_bpSaved'  ,'w_1stIn' ,'w_1stWon' ,'w_SvGms' ,'w_bpSaved'],axis=1)

#create subsets according to target variable
df_five = df[(df['best_of']== 5) & (df['set'] >= 3)]
df_three = df[(df['best_of']== 3) & (df['set'] < 4)]
#print(df_five.shape)
#print(df_three.shape)

# data exploration and analytics

print(df_three['set'].describe())
print(df_five['set'].describe())


def barchart_1 (d):
    ax = d['set'].value_counts().plot(kind ="bar", figsize=(11,10), width = 0.8, color = 'grey', edgecolor='black')
    max = d['best_of'].max()
    ax.set_title(f"Number of Sets - max {max} sets", fontsize=18)
    ax.set_xlabel("Sets", fontsize = 15)
    ax.set_ylabel("Matches", fontsize =15)
    totals = []
    for i in ax.patches:
        totals.append(i.get_height())
    total = sum(totals)
    for i in ax.patches:
        ax.text(i.get_x()+.3, i.get_height()+ 40, \
            str(round((i.get_height()/total)*100, 2))+'%', fontsize=13,
                color='black')
    plt.show()

#barchart_1 (df_three)

#validate 2 sets % using binning

def stackedbar_1 (d):

    d['bins'] = pd.qcut(d['tourney_date'], q = 10, precision = 0)

    cross_tab_perc = pd.crosstab(index=d['bins'], columns=d['set'], normalize="index")

    print(cross_tab_perc)

    ax = cross_tab_perc.plot(kind='bar', stacked=True, colormap='Set1', figsize=(12, 8))
    ax.set_xlabel("Bins", fontsize =15)
    ax.set_ylabel("Matches %", fontsize =15)
    ax.set_yticklabels(['0%','20%', '40%', '60%', '80%', '100%'])
    ax.legend(loc='upper right', title="Number of Sets")
    ax.set_title ("Final sets percentage over 10 periods", fontsize =18)
    for c in ax.containers:
        ax.bar_label(c,label_type='center', fmt='%.2f')

    print(f"standard error = {cross_tab_perc[2].std() * 100}%")
    plt.show()

#stackedbar_1(df_three)


trials = df_three['set'].count()
success = df_three['set'][(df_three['set']== 2)].count()
print(binomtest(success,trials,0.62,'greater'))

#print(df_three.info())
category = ['loser_hand', 'tourney_level', 'winner_hand', 'round',  'surface', 'underdog_win']
for x in category:
    print('Number of Sets, Correlation by:', x)
    print(pd.crosstab(index=df_three[x], columns=df_three['set'],normalize="index"))
    print(pd.crosstab(index=df_three[x], columns=df_three['set']))
    print('-'*15, '\n')

# explore w/l hand

def expHands(d):
    cross_tab_WH = pd.crosstab(index=d['winner_hand'][d['winner_hand']!= 'U'], columns=d['set'], normalize="index")
    cross_tab_LH = pd.crosstab(index=d['loser_hand'][d['loser_hand']!= 'U'], columns=d['set'], normalize="index")

    fig,axes = plt.subplots(1,2, figsize = (14,8), sharey = True)
    cross_tab_LH.plot(kind='bar', stacked=True, colormap='Set1',figsize = (10,5), ax = axes[0])
    axes[0].set_xlabel("Loser Hand", fontsize =15)
    axes[0].set_ylabel("Matches %", fontsize =15)
    axes[0].set_yticklabels(['0%','20%', '40%', '60%', '80%', '100%'])
    axes[0].legend(loc='upper right', title="Number of Sets")
    for c in axes[0].containers:
        axes[0].bar_label(c, label_type='center', fmt='%.2f')

    cross_tab_WH.plot(kind='bar', stacked=True, colormap='Set1',figsize = (10,5), ax = axes[1])
    axes[1].set_xlabel("Winner Hand", fontsize =15)
    axes[1].set_ylabel("Matches %", fontsize =15)
    axes[1].legend(loc='upper right', title="Number of Sets")
    for c in axes[1].containers:
        axes[1].bar_label(c, label_type='center', fmt='%.2f')
    plt.show()

#expHands (df_three)


# surface

def PCsurface (d):
    d['Matches'] = 1

    fig, axes = plt.subplots(1,3,figsize=(15, 8))
    fig.suptitle('Number of sets % on different surfaces')
    d[(d['surface']=='Clay')].groupby(d['set']).count().plot\
        (kind='pie', y = 'Matches', autopct='%1.2f%%', ax = axes[0])
    axes[0].legend(loc='upper right', title="Number of Sets")
    axes[0].set_xlabel("Clay", fontsize =15)
    axes[0].set(ylabel=None)
    d[(d['surface']=='Grass')].groupby(d['set']).count().plot\
        (kind='pie', y='Matches', autopct='%1.2f%%', ax = axes[1])
    axes[1].set_xlabel("Grass", fontsize =15)
    axes[1].set(ylabel=None)
    axes[1].legend(loc='upper right', title="Number of Sets")
    d[(d['surface']=='Hard')].groupby(d['set']).count().plot\
        (kind='pie', y='Matches', autopct='%1.2f%%', ax = axes[2])
    axes[2].set_xlabel("Hard", fontsize =15)
    axes[2].set(ylabel=None)
    axes[2].legend(loc='upper right', title="Number of Sets")
    plt.show()

#PCsurface(df_three)

# explore round
def ExpRound (d):
    cross_tab_round = pd.crosstab(index=d['round'][d['round']!= 'BR'], columns=d['set'], normalize="index")

    ax = cross_tab_round.plot(kind='bar', stacked=True, colormap='Set1',figsize = (10,5))
    ax.set_xlabel("Round", fontsize =15)
    ax.set_ylabel("Matches %", fontsize =15)
    ax.set_yticklabels(['0%','20%', '40%', '60%', '80%', '100%'])
    ax.legend(loc='upper right', title="Number of Sets")
    ax.set_title("Number of sets % in different rounds")
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fmt='%.2f')
    plt.show()

#ExpRound(df_three)

# hypothesis testing round
trials = df_three['round'][(df_three['round']=='RR')].count()
success = df_three['set'][(df_three['set']== 2) & (df_three['round']=='RR')].count()
print(binomtest(success,trials,0.6,'less'))

trials = df_three['round'][(df_three['round']=='R128')].count()
success = df_three['set'][(df_three['set']== 2) & (df_three['round']=='R128')].count()
print(binomtest(success,trials,0.6,'less'))

# explore level
def ExpLevel (d):
    cross_tab_round = pd.crosstab(index=d['tourney_level'][d['tourney_level']!= 'F'], columns=d['set'], normalize="index")

    ax = cross_tab_round.plot(kind='bar', stacked=True, colormap='Set1',figsize = (10,5))
    ax.set_xlabel("Tournament Level", fontsize =15)
    ax.set_ylabel("Matches %", fontsize =15)
    ax.set_yticklabels(['0%','20%', '40%', '60%', '80%', '100%'])
    ax.legend(loc='upper right', title="Number of Sets")
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fmt='%.2f')
    plt.show()

#ExpLevel(df_three)

#testing Tournament Level

trials = df_three['tourney_level'][(df_three['tourney_level']=='D')].count()
success = df_three['set'][(df_three['set']== 2) & (df_three['tourney_level']=='D')].count()
#print(binomtest(success,trials,0.6,'less'))

#exploring underdog_win
def ExpUdog (d):
    cross_tab_round = pd.crosstab(index=d['underdog_win'], columns=d['set'], normalize="index")

    ax = cross_tab_round.plot(kind='bar', stacked=True, colormap='Set1',figsize = (10,5))
    ax.set_xlabel("UnderDog Win", fontsize =15)
    ax.set_ylabel("Matches %", fontsize =15)
    ax.set_yticklabels(['0%','20%', '40%', '60%', '80%', '100%'])
    ax.legend(loc='upper right', title="Number of Sets")
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fmt='%.2f')
    plt.show()

#ExpUdog(df_three)

#testing Underdog
trials = df_three['underdog_win'][(df_three['underdog_win']=='yes')].count()
success = df_three['set'][(df_three['set']== 2) & (df_three['underdog_win']=='yes')].count()
print(binomtest(success,trials,0.59,'less'))
print(binomtest(success,trials,0.6,'less'))

trials = df_three['underdog_win'][(df_three['underdog_win']=='no')].count()
success = df_three['set'][(df_three['set']== 2) & (df_three['underdog_win']=='no')].count()
print(binomtest(success,trials,0.64,'greater'))
print(binomtest(success,trials,0.65,'greater'))

# numerical variables

# create bins

num_var = ['l_2ndWon','l_ace','l_bpFaced','l_df','l_svpt','loser_age','loser_rank','loser_rank_points',\
    'w_2ndWon','w_ace','w_bpFaced','w_df','w_svpt','winner_age','winner_rank','winner_rank_points','rank_diff']

df_bins = df_three.loc[:,num_var]

for names,values in df_bins.iteritems():
    df_bins[names] = pd.qcut(values, 7, duplicates='drop')

df_bins['set'] = df_three['set']
#print(df_bins.info())

#loser match stats

plt.figure(figsize=[9,15])

plt.subplot(151)
dfbg = df_bins[(df_bins['set']==2)].groupby(['l_ace'])['set'].count() / df_bins.groupby(['l_ace'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Loser Aces')
plt.xlabel('Loser Aces bins', fontsize=15)
plt.ylabel('% of 2 set matches', fontsize=15)
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(152)
dfbg = df_bins[(df_bins['set']==2)].groupby(['l_2ndWon'])['set'].count() / df_bins.\
    groupby(['l_2ndWon'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Loser 2nd Won')
plt.xlabel('Loser 2nd Won bins', fontsize=15)
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(153)
dfbg = df_bins[(df_bins['set']==2)].groupby(['l_bpFaced'])['set'].count() / df_bins.\
    groupby(['l_bpFaced'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Loser BP faced')
plt.xlabel('Loser BP faced bins', fontsize=15)
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(154)
dfbg = df_bins[(df_bins['set']==2)].groupby(['l_df'])['set'].count() / df_bins.\
    groupby(['l_df'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Loser double faults')
plt.xlabel('Loser double faults bins', fontsize=15)
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(155)
dfbg = df_bins[(df_bins['set']==2)].groupby(['l_svpt'])['set'].count() / df_bins.\
    groupby(['l_svpt'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Loser service points')
plt.xlabel('Loser service points bins', fontsize=15)
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

#winner match stats

plt.figure(figsize=[8,15])

plt.subplot(151)
dfbg = df_bins[(df_bins['set']==2)].groupby(['w_ace'])['set'].count() / df_bins.groupby(['w_ace'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Winner Aces bins')
plt.ylabel('% of 2 set matches', fontsize=15)
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(152)
dfbg = df_bins[(df_bins['set']==2)].groupby(['w_2ndWon'])['set'].count() / df_bins.\
    groupby(['w_2ndWon'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Winner 2nd Won bins')
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(153)
dfbg = df_bins[(df_bins['set']==2)].groupby(['w_bpFaced'])['set'].count() / df_bins.\
    groupby(['w_bpFaced'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Winner BP faced bins')
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(154)
dfbg = df_bins[(df_bins['set']==2)].groupby(['w_df'])['set'].count() / df_bins.\
    groupby(['w_df'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Winner double faults bins')
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(155)
dfbg = df_bins[(df_bins['set']==2)].groupby(['w_svpt'])['set'].count() / df_bins.\
    groupby(['w_svpt'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Winner service points bins')
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.figure(figsize=[8,15])

plt.subplot(131)
dfbg = df_bins[(df_bins['set']==2)].groupby(['loser_age'])['set'].count() / df_bins.groupby(['loser_age'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Loser Age bins')
plt.ylabel('% of 2 set matches', fontsize=15)
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(132)
dfbg = df_bins[(df_bins['set']==2)].groupby(['loser_rank'])['set'].count() / df_bins.\
    groupby(['loser_rank'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Loser Rank bins')
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(133)
dfbg = df_bins[(df_bins['set']==2)].groupby(['loser_rank_points'])['set'].count() / df_bins.\
    groupby(['loser_rank_points'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Loser Rank Points bins')
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.figure(figsize=[8,15])
plt.subplot(131)
ax = dfbg = df_bins[(df_bins['set']==2)].groupby(['winner_age'])['set'].count() / df_bins.\
    groupby(['winner_age'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Winner Age bins')
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(132)
dfbg = df_bins[(df_bins['set']==2)].groupby(['winner_rank'])['set'].count() / df_bins.\
    groupby(['winner_rank'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Winner Rank bins')
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.subplot(133)
dfbg = df_bins[(df_bins['set']==2)].groupby(['winner_rank_points'])['set'].count() / df_bins.\
    groupby(['winner_rank_points'])['set'].count()
ax = dfbg.plot(kind='bar',  figsize=(10, 7), alpha = 0.8, width = 0.8)
plt.title('Winner rank points bins')
yticks = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in yticks])

plt.show()

#loser aces 'l_ace'

print(df_three['l_ace'].describe())
print(pd.crosstab(index=df_bins['l_ace'], columns=df_bins['set'], normalize="index"))
df_bins['l_ace'] = pd.cut(df_three['l_ace'], 20)
plt.hist(df_three['l_ace'], bins = 20)
plt.title('Distribution of loser aces')
plt.show()

title = 'Loser Aces Bins'
def StackBar2 (i,c):
    cross_tab_RF = pd.crosstab(index=i, columns=c, normalize="index")
    cross_tab_RF2 = pd.crosstab(index=i, columns=c)
    ax = cross_tab_RF.plot(kind='bar', stacked=True, colormap='Set1',figsize = (10,5))
    ax.set_ylabel("Matches %", fontsize =15)
    ax.set_yticklabels(['0%','20%', '40%', '60%', '80%', '100%'])
    ax.legend(loc='upper right', title="Number of Sets")
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fmt='%.2f')
    plt.title(title)
    plt.show()
    print(cross_tab_RF)
    print(cross_tab_RF2)

StackBar2(df_bins['l_ace'], df_bins['set'])
df_bins['l_ace'] = pd.cut(df_three['l_ace'], bins = [-0.001, 1, 2, 3, 4, 7, 20])
StackBar2(df_bins['l_ace'], df_bins['set'])

#test loser aces < 1
trials = df_three['l_ace'][(df_three['l_ace']<1.01)].count()
success = df_three['l_ace'][(df_three['set']== 2) & (df_three['l_ace']<1.1)].count()
print(binomtest(success,trials,0.70,'greater'))

#loser service points

#print(df_three['l_svpt'].describe())
#print(pd.crosstab(index=df_bins['l_svpt'], columns=df_bins['set'], normalize="index"))
df_bins['l_svpt'] = pd.cut(df_three['l_svpt'], 30)
plt.hist(df_three['l_svpt'], bins = 20)
plt.title('Distribution of loser service points')
plt.show()

title = 'Loser Service Points Bins'
def StackBar2 (i,c):
    cross_tab_RF = pd.crosstab(index=i, columns=c, normalize="index")
    cross_tab_RF2 = pd.crosstab(index=i, columns=c)
    ax = cross_tab_RF.plot(kind='bar', stacked=True, colormap='Set1',figsize = (10,5))
    ax.set_ylabel("Matches %", fontsize =15)
    ax.set_yticklabels(['0%','20%', '40%', '60%', '80%', '100%'])
    ax.legend(loc='upper right', title="Number of Sets")
    for c in ax.containers:
        ax.bar_label(c, label_type='center', fmt='%.2f')
    plt.title(title)
    plt.show()
    print(cross_tab_RF)
    print(cross_tab_RF2)

StackBar2(df_bins['l_svpt'], df_bins['set'])
df_bins['l_svpt'] = pd.cut(df_three['l_svpt'], bins = [-0.001, 25 , 42,  100])
StackBar2(df_bins['l_svpt'], df_bins['set'])

#test loser svpt <= 25
trials = df_three['l_svpt'][(df_three['l_svpt']<=25)].count()
success = df_three['l_svpt'][(df_three['set']== 2) & (df_three['l_svpt']<=25)].count()
print(binomtest(success,trials,0.73,'greater'))

#test loser svpt >= 42
trials = df_three['l_svpt'][(df_three['l_svpt']>=42)].count()
success = df_three['l_svpt'][(df_three['set']== 2) & (df_three['l_svpt']>=42)].count()
#print(binomtest(success,trials,0.71,'greater'))

#rank point difference

print(pd.crosstab(index=df_bins['rank_diff'], columns=df_bins['set'], normalize="index"))

df_bins['rank_diff'] = pd.cut(df_three['rank_diff'], 20)
#plt.hist(df_three['rank_diff'], bins = 20)
#plt.title('Distribution of rank points difference')
#plt.show()


title = 'Rank Difference Bins'
#StackBar2(df_bins['rank_diff'], df_bins['set'])

df_bins['rank_diff'] = pd.cut(df_three['rank_diff'], bins = [0, 2000, 3500 ,5500, 9000, 17000 ])

#StackBar2(df_bins['rank_diff'], df_bins['set'])

trials = df_three['rank_diff'][(df_three['rank_diff']>=3500)].count()
success = df_three['rank_diff'][(df_three['set']== 2) & (df_three['rank_diff']>=3500)].count()
#print(binomtest(success,trials,0.66,'greater'))

belowTHR = df_three['rank_diff'][(df_three['rank_diff']<3500)].count()
y = np.array([belowTHR,trials])
labels = ['under 3500 pts', 'over 3500 pts']
#plt.pie(y, labels = labels, startangle = 90, autopct='%1.2f%%')
#plt.title('% of matches with difference under/over 3500 points')
#plt.show()

#winner double faults

#print(df_three['w_df'].describe())
#print(pd.crosstab(index=df_bins['w_df'], columns=df_bins['set'], normalize="index"))
df_bins['w_df'] = pd.cut(df_three['w_df'], bins = [-0.001, 1, 2, 3, 4, 6])
title = 'Winner Double Faults Bins'
#StackBar2(df_bins['w_df'], df_bins['set'])

# winner rank
print(pd.crosstab(index=df_bins['winner_rank'], columns=df_bins['set'], normalize="index"))
print(pd.crosstab(index=df_bins['winner_rank'], columns=df_bins['set']))
df_bins['winner_rank'] = pd.cut(df_three['winner_rank'], bins = [-0.001, 10, 100, 200, 2000])
title = 'Winner rank'
StackBar2(df_bins['winner_rank'], df_bins['set'])

trials = df_three['winner_rank'][(df_three['winner_rank']<11)].count()
success = df_three['winner_rank'][(df_three['set']== 2) & (df_three['winner_rank']<11)].count()
print(binomtest(success,trials,0.68,'greater'))

trials = df_three['winner_rank'][(df_three['winner_rank']>100)].count()
success = df_three['winner_rank'][(df_three['set']== 2) & (df_three['winner_rank']>100)].count()
print(binomtest(success,trials,0.59,'less'))

# loser rank
print(pd.crosstab(index=df_bins['loser_rank'], columns=df_bins['set'], normalize="index"))
print(pd.crosstab(index=df_bins['loser_rank'], columns=df_bins['set']))
title = 'Loser rank'
StackBar2(df_bins['loser_rank'], df_bins['set'])
df_bins['loser_rank'] = pd.cut(df_three['loser_rank'], bins = [-0.001, 10, 20, 100, 150, 200, 2000])
StackBar2(df_bins['loser_rank'], df_bins['set'])

trials = df_three['loser_rank'][(df_three['loser_rank']>100)].count()
success = df_three['loser_rank'][(df_three['set']== 2) & (df_three['loser_rank']>100)].count()
print(binomtest(success,trials,0.64,'greater'))

