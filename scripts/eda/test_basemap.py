import os, math, urllib.request
from PIL import Image
import numpy as np

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_basemap(lon_min, lon_max, lat_min, lat_max, zoom=13, cache_dir="artifacts/cache/tiles"):
    os.makedirs(cache_dir, exist_ok=True)
    x_min, y_min = deg2num(lat_max, lon_min, zoom)
    x_max, y_max = deg2num(lat_min, lon_max, zoom)
    
    print(f"Fetching tiles x: {x_min}..{x_max}, y: {y_min}..{y_max} (zoom {zoom})")
    n_x = x_max - x_min + 1
    n_y = y_max - y_min + 1
    
    stitched = Image.new("RGB", (n_x * 256, n_y * 256))
    
    for i, x in enumerate(range(x_min, x_max + 1)):
        for j, y in enumerate(range(y_min, y_max + 1)):
            cache_file = os.path.join(cache_dir, f"{zoom}_{x}_{y}.png")
            if not os.path.exists(cache_file):
                url = f"https://a.basemaps.cartocdn.com/light_all/{zoom}/{x}/{y}.png"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                try:
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        with open(cache_file, "wb") as f_out:
                            f_out.write(resp.read())
                except Exception as e:
                    print(f"Failed to fetch tile {x},{y}: {e}")
                    tile = Image.new("RGB", (256, 256), (245, 245, 245))
                    stitched.paste(tile, (i * 256, j * 256))
                    continue
            tile = Image.open(cache_file)
            stitched.paste(tile, (i * 256, j * 256))
            
    # Calculate exact bounds of stitched image
    top_lat, left_lon = num2deg(x_min, y_min, zoom)
    bot_lat, right_lon = num2deg(x_max + 1, y_max + 1, zoom)
    
    return np.array(stitched), [left_lon, right_lon, bot_lat, top_lat]

if __name__ == "__main__":
    img, extent = fetch_basemap(-1.66, -1.57, 54.96, 55.01, zoom=13)
    print("Fetched basemap image shape:", img.shape, "extent:", extent)
