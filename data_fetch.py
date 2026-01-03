import math
import time
import requests
from PIL import Image
from io import BytesIO

ZOOM = 19
TILE_SIZE = 256
TILES_AROUND = 1  # 3x3 grid

HEADERS = {
    "User-Agent": "HousePriceMultimodalProject/1.0 (educational)"
}

def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
        / 2.0 * n
    )
    return xtile, ytile


def fetch_satellite_image(lat, lon, save_path):
    x, y = latlon_to_tile(lat, lon, ZOOM)

    img_size = TILE_SIZE * (2 * TILES_AROUND + 1)
    final_img = Image.new("RGB", (img_size, img_size))

    for dx in range(-TILES_AROUND, TILES_AROUND + 1):
        for dy in range(-TILES_AROUND, TILES_AROUND + 1):
            tile_x = x + dx
            tile_y = y + dy

            url = (
                f"https://services.arcgisonline.com/ArcGIS/rest/services/"
                f"World_Imagery/MapServer/tile/{ZOOM}/{tile_y}/{tile_x}"
            )

            r = requests.get(url, headers=HEADERS, timeout=10)
            tile_img = Image.open(BytesIO(r.content)).convert("RGB")

            px = (dx + TILES_AROUND) * TILE_SIZE
            py = (dy + TILES_AROUND) * TILE_SIZE
            final_img.paste(tile_img, (px, py))

    final_img.save(save_path)
