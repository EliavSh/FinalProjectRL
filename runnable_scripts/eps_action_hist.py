import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

resolution = 100000

output = pd.read_excel('C:/Users/ELIAV/Google Drive/Final Project/FinalProjectRL/results/results_07_08_2020_00.xlsx')

data = pd.DataFrame(
    data=np.transpose(np.array([list(map(lambda x: x // resolution, np.array(list(output.index)))), np.array(output['action']), np.ones(len(output))])),
    columns=['step', 'action', 'sum']).groupby(['step', 'action']).sum().reset_index()

rows = zip(data['step'] * resolution + resolution, data['action'], data['sum'])
headers = ['step', 'action', 'sum']
df = pd.DataFrame(rows, columns=headers)

fig, ax = plt.subplots(figsize=(10, 7))

months = df['action'].drop_duplicates()
margin_bottom = np.zeros(len(df['step'].drop_duplicates()))
colors = ["#006D2C", "#31A354", "#74C476"]

for num, month in enumerate(months):
    values = list(df[df['action'] == month].loc[:, 'sum'])

    df[df['action'] == month].plot.bar(x='step', y='sum', ax=ax, stacked=True, bottom=margin_bottom, color=colors[num], label=month)
    margin_bottom += values

plt.show()
print('eliav king')
