#!/usr/bin/env python3
"""OpenStreetMap places + building footprints -> <world>_net/pois.yaml (the map's POI layer).

  python3 tools/osm_pois.py worlds/osm_source/metro_city.osm worlds/metro_city_net

Same projection and centring as tools/osm_to_net.py / the Webots OSM importer, so places line up
with the 3D buildings and the lane map. Kinds: hospital, clinic, pharmacy, police, fire_station,
fuel, school, bank, pub, bar, nightclub, restaurant, cafe, fast_food, shop, hotel, place_of_worship,
parking, bus_stop, plus every named/unnamed building footprint (kind 'building').
"""
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import pyproj
import yaml

AMENITY = {'hospital': 'hospital', 'clinic': 'clinic', 'doctors': 'clinic', 'pharmacy': 'pharmacy',
           'police': 'police', 'fire_station': 'fire_station', 'fuel': 'fuel', 'school': 'school',
           'college': 'school', 'university': 'school', 'bank': 'bank', 'pub': 'pub', 'bar': 'bar',
           'nightclub': 'nightclub', 'restaurant': 'restaurant', 'cafe': 'cafe', 'fast_food': 'fast_food',
           'food_court': 'food_court', 'place_of_worship': 'place_of_worship', 'parking': 'parking',
           'cinema': 'cinema', 'theatre': 'theatre', 'library': 'library'}


def main():
    osm, out_dir = sys.argv[1], sys.argv[2]
    head = open(osm).read(4000)
    b = re.search(r'<bounds minlat="([-\d.]+)" minlon="([-\d.]+)" maxlat="([-\d.]+)" maxlon="([-\d.]+)"', head)
    minlat, minlon, maxlat, maxlon = (float(v) for v in b.groups())
    lat0, lon0 = 0.5 * (minlat + maxlat), 0.5 * (minlon + maxlon)
    zone = 1 + math.floor((lon0 + 180) / 6)
    P = pyproj.Proj(f'+proj=utm +north +zone={zone} +lon_0={lon0:f} +lat_0={lat0:f} +x_0=0 +y_0=0 '
                    f'+ellps=WGS84 +units=m +no_defs')
    x1, y1 = P(minlon, minlat)
    x2, y2 = P(maxlon, maxlat)
    xo, yo = (x1 + x2) / 2, (y1 + y2) / 2

    def xy(lon, lat):
        x, y = P(lon, lat)
        return round(x - xo, 2), round(y - yo, 2)

    def clean(name):
        # generic city: no real city name in place labels ("<City> General Hospital" -> "General Hospital")
        name = re.sub(r'\s*\bSingapore\b\s*', ' ', name or '').strip(' -,')
        return re.sub(r'\s{2,}', ' ', name)

    root = ET.parse(osm).getroot()
    nodes = {n.get('id'): (float(n.get('lon')), float(n.get('lat'))) for n in root.iter('node')}
    places = []

    def kind_of(tags):
        if tags.get('amenity') in AMENITY:
            return AMENITY[tags['amenity']]
        if tags.get('healthcare') in ('hospital', 'clinic'):
            return tags['healthcare']
        if 'shop' in tags:
            return 'shop'
        if tags.get('tourism') in ('hotel', 'hostel'):
            return 'hotel'
        if tags.get('highway') == 'bus_stop':
            return 'bus_stop'
        if tags.get('building') in ('hospital',):
            return 'hospital'
        return None

    for n in root.iter('node'):
        tags = {t.get('k'): t.get('v') for t in n.iter('tag')}
        k = kind_of(tags)
        if k:
            x, y = xy(float(n.get('lon')), float(n.get('lat')))
            places.append({'name': clean(tags.get('name:en') or tags.get('name')) or k.replace('_', ' ').title(),
                           'kind': k, 'x': x, 'y': y, 'height': 4.0,
                           'footprint': [[x - 2, y - 2], [x + 2, y - 2], [x + 2, y + 2], [x - 2, y + 2]]})
    for w in root.iter('way'):
        tags = {t.get('k'): t.get('v') for t in w.iter('tag')}
        if 'building' not in tags and kind_of(tags) is None:
            continue
        refs = [nd.get('ref') for nd in w.iter('nd') if nd.get('ref') in nodes]
        if len(refs) < 3:
            continue
        foot = [list(xy(*nodes[r])) for r in refs[:-1] if r in nodes]
        cx, cy = sum(p[0] for p in foot) / len(foot), sum(p[1] for p in foot) / len(foot)
        levels = tags.get('building:levels', '')
        h = float(tags['height']) if re.fullmatch(r'[\d.]+', tags.get('height', '')) else (
            3.0 * float(levels) if re.fullmatch(r'[\d.]+', levels) else 10.0)
        k = kind_of(tags) or 'building'
        name = clean(tags.get('name:en') or tags.get('name') or '')
        places.append({'name': name or ('Building' if k == 'building' else k.replace('_', ' ').title()),
                       'kind': k, 'x': round(cx, 2), 'y': round(cy, 2), 'height': round(h, 1), 'footprint': foot})
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'pois.yaml')
    yaml.safe_dump({'source': os.path.basename(osm), 'places': places}, open(out, 'w'), sort_keys=False)
    kinds = {}
    for p in places:
        kinds[p['kind']] = kinds.get(p['kind'], 0) + 1
    print(f'{len(places)} places -> {out}')
    print(dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    for p in places:
        if p['kind'] in ('hospital', 'nightclub', 'pub', 'police') and p['name']:
            print(f"  {p['kind']:10s} {p['name'][:45]:45s} ({p['x']:.0f}, {p['y']:.0f})")


if __name__ == '__main__':
    main()
