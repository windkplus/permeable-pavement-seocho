"""
01_gis_preprocessing.py
서초구 행정동별 소유역 파라미터 산출
  - 면적 (ha)
  - 불투수율 (%) ← OSM 건물+도로 면적 기반
  - 평균경사도 (degree) ← Copernicus DEM 기반
출력: results/subcatchment_params.csv
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOUNDARY_SHP = os.path.join(BASE_DIR, 'data/boundary/bnd_dong_11220_2025_2Q/bnd_dong_11220_2025_2Q.shp')
DEM_E126     = os.path.join(BASE_DIR, 'data/dem/Copernicus_DSM_COG_10_N37_00_E126_00_DEM.tif')
DEM_E127     = os.path.join(BASE_DIR, 'data/dem/Copernicus_DSM_COG_10_N37_00_E127_00_DEM.tif')
DEM_MERGED   = os.path.join(BASE_DIR, 'data/dem/dem_merged_5179.tif')
OUTPUT_CSV   = os.path.join(BASE_DIR, 'results/subcatchment_params.csv')

os.makedirs(os.path.join(BASE_DIR, 'results'), exist_ok=True)

TARGET_CRS = 'EPSG:5179'

# ── 1. 행정동 경계 로드 ────────────────────────────────────────────────────────
print("1. 행정동 경계 로드 중...")
dong = gpd.read_file(BOUNDARY_SHP).to_crs(TARGET_CRS)
print(f"   {len(dong)}개 행정동 로드 완료")

# ── 2. DEM 두 타일 병합 및 재투영 ─────────────────────────────────────────────
if not os.path.exists(DEM_MERGED):
    print("2. DEM 타일 병합 및 재투영 중 (최초 1회)...")
    src_files = []
    for path in [DEM_E126, DEM_E127]:
        if os.path.exists(path):
            src_files.append(rasterio.open(path))

    mosaic, out_transform = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update({"driver": "GTiff", "height": mosaic.shape[1],
                     "width": mosaic.shape[2], "transform": out_transform})

    tmp_path = os.path.join(BASE_DIR, 'data/dem/dem_merged_wgs84.tif')
    with rasterio.open(tmp_path, 'w', **out_meta) as tmp:
        tmp.write(mosaic)
    for s in src_files:
        s.close()

    # WGS84 → EPSG:5179 재투영
    with rasterio.open(tmp_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': TARGET_CRS, 'transform': transform,
                       'width': width, 'height': height, 'driver': 'GTiff'})
        with rasterio.open(DEM_MERGED, 'w', **kwargs) as dst:
            reproject(source=rasterio.band(src, 1),
                      destination=rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=TARGET_CRS,
                      resampling=Resampling.bilinear)
    os.remove(tmp_path)
    print("   DEM 병합 완료")
else:
    print("2. 병합된 DEM 이미 존재, 재사용")

# ── 3. OSM 건물+도로 다운로드 (서초구 전체) ────────────────────────────────────
print("3. OSM 데이터 다운로드 중 (인터넷 연결 필요)...")
place = "서초구, 서울특별시, 대한민국"

buildings = ox.features_from_place(place, tags={"building": True})
buildings = buildings[buildings.geometry.geom_type.isin(
    ['Polygon', 'MultiPolygon'])].to_crs(TARGET_CRS)
print(f"   건물 {len(buildings)}개 다운로드 완료")

road_types = ["motorway","trunk","primary","secondary","tertiary",
              "residential","service","unclassified"]
roads = ox.features_from_place(place, tags={"highway": road_types})
roads = roads[roads.geometry.geom_type.isin(
    ['LineString', 'MultiLineString'])].to_crs(TARGET_CRS)

# 도로 폭 기준 버퍼 (도로 등급별)
road_width = {"motorway": 15, "trunk": 12, "primary": 10, "secondary": 8,
              "tertiary": 6, "residential": 4, "service": 3, "unclassified": 4}
roads['width'] = roads['highway'].map(
    lambda x: road_width.get(x[0] if isinstance(x, list) else x, 4))
roads_buf = roads.copy()
roads_buf['geometry'] = roads.geometry.buffer(roads['width'] / 2)
print(f"   도로 {len(roads)}개 다운로드 완료")

# ── 4. 소유역별 파라미터 계산 ──────────────────────────────────────────────────
print("4. 소유역별 파라미터 계산 중...")
results = []

with rasterio.open(DEM_MERGED) as dem_ds:
    pixel_m = dem_ds.res[0]
    nodata  = dem_ds.nodata

    for _, row in dong.iterrows():
        dong_name = row['ADM_NM']
        dong_code = row['ADM_CD']
        geom      = row.geometry

        # 면적 (ha)
        area_m2 = geom.area
        area_ha = area_m2 / 10_000

        # 불투수율: (건물면적 + 도로면적) / 동면적
        bld_clip  = buildings[buildings.geometry.intersects(geom)]
        road_clip = roads_buf[roads_buf.geometry.intersects(geom)]

        bld_area  = bld_clip.geometry.intersection(geom).area.sum()
        road_area = road_clip.geometry.intersection(geom).area.sum()
        imp_area  = min(bld_area + road_area, area_m2)   # 100% 초과 방지
        impervious_pct = imp_area / area_m2 * 100

        # 평균경사도
        try:
            arr, _ = mask(dem_ds, [mapping(geom)], crop=True)
            arr = arr[0].astype(float)
            if nodata is not None:
                arr[arr == nodata] = np.nan
            dy, dx = np.gradient(arr, pixel_m)
            slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            avg_slope = float(np.nanmean(slope_deg))
        except Exception:
            avg_slope = np.nan

        results.append({
            'ADM_CD'        : dong_code,
            'ADM_NM'        : dong_name,
            'area_ha'       : round(area_ha, 2),
            'impervious_pct': round(impervious_pct, 1),
            'avg_slope_deg' : round(avg_slope, 2) if not np.isnan(avg_slope) else None
        })
        print(f"   {dong_name:8s} | 면적={area_ha:7.1f}ha | "
              f"불투수율={impervious_pct:5.1f}% | 경사={avg_slope:.2f}°")

# ── 5. 결과 저장 ───────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

print(f"\n완료! → {OUTPUT_CSV}")
print("\n=== 서초구 소유역 파라미터 요약 ===")
print(df.to_string(index=False))
print(f"\n서초구 전체 면적 : {df['area_ha'].sum():.1f} ha")
print(f"평균 불투수율    : {df['impervious_pct'].mean():.1f} %")
print(f"평균 경사도      : {df['avg_slope_deg'].mean():.2f} °")
