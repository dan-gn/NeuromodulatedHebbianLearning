from scipy import stats
import pandas as pd
data = pd.read_csv('final_results_gecco_april2.csv')

# env = 'CartPole-v1'
# env = 'Acrobot-v1'
env = 'MountainCar-v0'

data = data[data['environment'] == env]

samples_fixed = data[data['model'] == 'static']['test']
samples_hl = data[data['model'] == 'abcd']['test']
samples_tnhl = data[data['model'] == 'neuromodulated_hb']['test']

print(samples_fixed.shape)
print(samples_hl.shape)
print(samples_tnhl.shape)


result = stats.kruskal(samples_fixed, samples_hl)
print(result.pvalue)
result = stats.kruskal(samples_hl, samples_tnhl)
print(result.pvalue)
result = stats.kruskal(samples_fixed, samples_tnhl)
print(result.pvalue)
result = stats.kruskal(samples_fixed, samples_hl, samples_tnhl)
print(result.pvalue)