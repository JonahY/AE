features_184  = pd.read_csv(r'C:\Users\Yuan\Desktop\AE\cls\20220225\Ni_features_184_184__4_GAP_Linear_origin256.csv')
data = features_184.to_numpy()

S_, A_ = ICA(3, *data.T)
km = KernelKMeans(n_clusters=2, max_iter=100, random_state=206, verbose=1, kernel="rbf")
pred = km.fit_predict(S_)
print(sum(pred))
cls_KKM = []
for i in range(2):
    cls_KKM.append(pred == i)
cls_KKM[0], cls_KKM[1] = pred == 1, pred == 0

fig = plt.figure(figsize=[6, 3.9])
ax = plt.subplot(projection='3d')
ax.scatter3D(S_[:, 0][cls_KKM[0]], S_[:, 1][cls_KKM[0]], S_[:, 2][cls_KKM[0]], s=15, color=color_1)
ax.scatter3D(S_[:, 0][cls_KKM[1]], S_[:, 1][cls_KKM[1]], S_[:, 2][cls_KKM[1]], s=15, color=color_2)
plot_norm(ax, '1st', '2nd', '', legend=False, frameon=False, fontname='Arial')
