from src.data_loader import load_bci_data

data_path = r"C:\Users\shari\Videos\nubrafusex\data\BCICIV_2a_gdf"
X, y, info = load_bci_data(data_path, tmin=0, tmax=4, verbose=True)
print(X.shape, y.shape)
