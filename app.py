from flask import Flask, render_template
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import os
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from shapely.geometry import Point
import geopandas as gpd
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ------------------------------
# 1) Generate synthetic data if CSV missing
def make_grid(center_lat=19.97, center_lon=73.75, n=1000):
    lats = center_lat + (np.random.rand(n)-0.5)*0.1
    lons = center_lon + (np.random.rand(n)-0.5)*0.12
    return lats, lons

def gen_rows(n=1000):
    rows=[]
    lats, lons = make_grid(n=n)
    for i in range(n):
        traffic = np.random.poisson(200)
        pop = max(50, int(np.random.normal(5000,2000)))
        commercial = np.clip(np.random.beta(2,5),0,1)
        ev = np.random.poisson(traffic*0.02 + commercial*10)
        existing = np.random.choice([0,0,0,1,2], p=[0.6,0.15,0.15,0.07,0.03])
        demand = (0.6*traffic/100 + 0.4*ev + commercial*5 + pop/1000 - existing*2) + np.random.normal(0,2)
        demand = max(0, demand)
        rows.append({
            'lat': lats[i],
            'lon': lons[i],
            'traffic_count': int(traffic),
            'ev_count': int(ev),
            'pop_density': float(pop),
            'commercial_index': float(commercial),
            'existing_chargers': int(existing),
            'target_demand': float(demand)
        })
    return pd.DataFrame(rows)

data_csv = 'data/ev_sample_data.csv'
if not os.path.exists(data_csv):
    df = gen_rows(1000)
    os.makedirs('data', exist_ok=True)
    df.to_csv(data_csv, index=False)
else:
    df = pd.read_csv(data_csv)

# ------------------------------
# 2) Aggregate into grid cells
def aggregate_grid(df, cell_size_m=1500):
    df['geometry'] = [Point(xy) for xy in zip(df.lon, df.lat)]
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
    gdf = gdf.to_crs(epsg=3857)
    minx, miny, maxx, maxy = gdf.total_bounds
    x_bins = np.arange(minx, maxx+cell_size_m, cell_size_m)
    y_bins = np.arange(miny, maxy+cell_size_m, cell_size_m)
    gdf['xbin'] = np.digitize(gdf.geometry.x, x_bins)-1
    gdf['ybin'] = np.digitize(gdf.geometry.y, y_bins)-1
    agg = gdf.groupby(['xbin','ybin']).agg({
        'traffic_count':'sum',
        'ev_count':'sum',
        'pop_density':'mean',
        'commercial_index':'mean',
        'existing_chargers':'sum',
        'target_demand':'sum'
    }).reset_index()
    xs = x_bins[agg.xbin] + cell_size_m/2
    ys = y_bins[agg.ybin] + cell_size_m/2
    gcent = gpd.GeoDataFrame(agg, geometry=gpd.points_from_xy(xs, ys, crs='EPSG:3857'))
    gcent = gcent.to_crs(epsg=4326)
    gcent['lat'] = gcent.geometry.y
    gcent['lon'] = gcent.geometry.x
    return gcent

grid = aggregate_grid(df)
grid = grid.dropna().reset_index(drop=True)

# ------------------------------
# 3) Unsupervised ML: K-Means clustering
coords = grid[['lat','lon']].values
n_clusters = min(12, max(2, int(len(grid)/20)))
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
grid['k_cluster'] = kmeans.fit_predict(coords)

# ------------------------------
# 4) Supervised ML: Random Forest regression (optional)
features = ['traffic_count','ev_count','pop_density','commercial_index','existing_chargers']
X = grid[features].fillna(0)
y = grid['target_demand'].fillna(0)
rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(X,y)
grid['pred_demand'] = rf.predict(X)
grid['priority_score'] = grid['pred_demand'] - grid['existing_chargers']*2
grid = grid.sort_values('priority_score', ascending=False).reset_index(drop=True)
grid['rank'] = grid.index + 1

top_n = min(100, len(grid))
top = grid.head(top_n)

# ------------------------------
# 5) Create Folium map
map_center = [top['lat'].mean(), top['lon'].mean()]
m = folium.Map(location=map_center, zoom_start=13)
marker_cluster = MarkerCluster().add_to(m)

for _, row in top.iterrows():
    radius = max(4, min(18, row['priority_score']/2)) if row['priority_score']>0 else 4
    folium.CircleMarker(
        location=[row['lat'], row['lon']],
        radius=radius,
        popup=f"Rank: {int(row['rank'])}<br>PredDemand: {row['pred_demand']:.2f}<br>Existing: {int(row['existing_chargers'])}",
        color='crimson',
        fill=True,
        fill_opacity=0.6
    ).add_to(marker_cluster)

os.makedirs('templates', exist_ok=True)
map_html_path = 'templates/map.html'
m.save(map_html_path)

# ------------------------------
# 6) Flask routes
@app.route('/')
def index():
    table_data = top[['rank','lat','lon','pred_demand','existing_chargers','priority_score']].round(3).to_dict(orient='records')
    return render_template('index.html', table_data=table_data, n_clusters=n_clusters)

@app.route('/map')
def map_view():
    return render_template('map.html')

# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
