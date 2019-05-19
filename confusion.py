import seaborn as sb
from sklearn.cluster import SpectralCoclustering

def list2pairs(l):
    pairs = list(itertools.combinations(l, 2))
    for i in l:
        pairs.append([i,i])
    return pairs
def ind(uniq,p):
    for i,v in enumerate(uniq):
        if p == v:
            return i
# get all genre lists pairs from all movies
allPairs = []
for i,row in data.iterrows():
    allPairs.extend(list2pairs(row['genre']))
n= len(uniq_pair)
uniq_pair = np.unique(allPairs)
grid = np.zeros((n, n))
for p in allPairs:
    grid[ind(uniq_pair==p[0]), ind(uniq_pair==p[1])]+=1
    if p[1] != p[0]:
        grid[ind(uniq_pair==p[1]), ind(uniq_pair==p[0])]+=1

print grid.shape
print len(Genre_ID_to_name.keys())

genre_list = []
sns.heatmap(grid, xticklabels=genre_list, yticklabels=genre_list)

model = SpectralCoclustering(n_clusters=5)
model.fit(grid)

fit_data = grid[np.argsort(model.row_labels_)]
fit_data = fit_data[:, np.argsort(model.column_labels_)]

genre_list_sorted = []
for i in np.argsort(model.row_labels_):
    genre_list_sorted.append(Genre_ID_to_name[uniq_pair[i]])

sb.heatmap(fit_data, xticklabels=genre_list_sorted, yticklabels=genre_list_sorted, annot=False)
plt.title("Biclustering")
plt.show()
