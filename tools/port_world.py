#!/usr/bin/env python3
"""Bring a Webots example world into this project, ready for the autonomy stack.

  python3 tools/port_world.py village_realistic --out village --start-lane -20_0 --start-s 25 \
      --hospital "building(1)"

Steps (all reversible: the original stays in the Webots install):
  1. copy <webots>/projects/vehicles/worlds/<name>.wbt and <name>_net/ to worlds/<out>.wbt, <out>_net/
  2. remove TrafficCone nodes (no cones on the road)
  3. add a SumoInterface (live SUMO traffic, which also reacts to our car) if the world has none
  4. traffic lights: GenericTrafficLight green 10 s / red 10 s
  5. place our car (DEF WEBOTS_VEHICLE0 BmwX5) on a lane of the lane map, facing along it, then
     fit the sensor suite with tools/install_sensors.py
  6. mark one building as the hospital: a red cross on its roof + name in worlds/<out>_net/pois.yaml
     (tools/extract_pois.py writes the full POI list: buildings, shops, restaurants, bus stops, ...)
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, '..'))
sys.path.insert(0, ROOT)


def remove_nodes(text, proto):
    out, n, i = [], 0, 0
    pat = re.compile(rf'^{proto} \{{', re.M)
    while True:
        m = pat.search(text, i)
        if not m:
            out.append(text[i:])
            break
        out.append(text[i:m.start()])
        depth, j = 0, m.end() - 1
        while True:
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        i = j + 2
        n += 1
    return ''.join(out), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('name')
    ap.add_argument('--out', required=True)
    ap.add_argument('--start-lane', required=True)
    ap.add_argument('--start-s', type=float, default=20.0)
    ap.add_argument('--hospital', default=None, help='name "..." of the building to mark as hospital')
    a = ap.parse_args()
    src_dir = os.path.join(os.environ.get('WEBOTS_HOME', os.path.expanduser('~/apps/webots')),
                           'projects', 'vehicles', 'worlds')
    wdir = os.path.join(ROOT, 'worlds')
    net_dir = os.path.join(wdir, f'{a.out}_net')
    shutil.copytree(os.path.join(src_dir, f'{a.name}_net'), net_dir, dirs_exist_ok=True)
    text = open(os.path.join(src_dir, f'{a.name}.wbt')).read()

    # SUMO spawns vehicles of these PROTOs: they must be declared IMPORTABLE, otherwise Webots
    # prints an error per spawn ("must be declared in the IMPORTABLE EXTERNPROTO list"), no traffic
    # appears and the error flood slowed the simulation to 15 % of real time.
    base = 'https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/vehicles/protos/'
    vehicles = ['bmw/BmwX5Simple', 'citroen/CitroenCZeroSimple', 'toyota/ToyotaPriusSimple', 'lincoln/LincolnMKZSimple',
                'range_rover/RangeRoverSportSVRSimple', 'tesla/TeslaModel3Simple',
                'mercedes_benz/MercedesBenzSprinterSimple', 'generic/BusSimple', 'generic/TruckSimple',
                'generic/TruckTrailerSimple', 'generic/TruckTankSimple', 'generic/ScooterSimple',
                'generic/ScooterDriver', 'generic/MotorbikeSimple', 'generic/MotorbikeDriver']
    for v in vehicles:   # a plain EXTERNPROTO is not enough for SUMO to spawn it: make it IMPORTABLE
        text = text.replace(f'\nEXTERNPROTO "{base}{v}.proto"', '')
    decl = ''.join(f'IMPORTABLE EXTERNPROTO "{base}{v}.proto"\n' for v in vehicles
                   if f'IMPORTABLE EXTERNPROTO "{base}{v}.proto"' not in text)
    text = text.replace('\nWorldInfo {', '\n' + decl + 'WorldInfo {', 1)
    text, n = remove_nodes(text, 'TrafficCone')
    print(f'removed {n} traffic cones')
    for proto in ('AdvertisingBoard', 'CyberboticsTower'):      # simulator-vendor branding
        text, n = remove_nodes(text, proto)
        print(f'removed {n} {proto}')
    if 'SumoInterface {' not in text:
        if 'SumoInterface.proto' not in text:
            text = text.replace('\nWorldInfo {', '\nEXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/'
                                'R2025a/projects/default/protos/SumoInterface.proto"\nWorldInfo {', 1)
        # at most 40 traffic cars, only within 200 m of our car: SUMO moves every car node each
        # step and Webots waits for it (100 cars across the whole map made the sim crawl)
        text += 'SumoInterface {\n  gui FALSE\n  maxVehicles 40\n  radius 200\n  enableHeight TRUE\n}\n'
        print('added SumoInterface (live traffic)')
    text = re.sub(r'(GenericTrafficLight \{\n(?:  [^\n]*\n)*?)\}', lambda m: re.sub(
        r'  (greenTime|redTime) [^\n]*\n', '', m.group(1)) + '  greenTime 10\n  redTime 10\n}', text)
    if 'BmwX5.proto"' not in text:
        text = text.replace('\nWorldInfo {', '\nEXTERNPROTO "https://raw.githubusercontent.com/cyberbotics/webots/'
                            'R2025a/projects/vehicles/protos/bmw/BmwX5.proto"\nWorldInfo {', 1)

    from auto_car_core.lanemap import LaneMap
    lm = LaneMap(os.path.join(net_dir, 'sumo.net.xml'))
    lane = lm.lanes[a.start_lane]
    p = lane.at(a.start_s)
    d = lane.at(a.start_s + 1.0) - p
    yaw = math.atan2(d[1], d[0])
    # BmwX5 origin is the rear axle: place it so the car sits on the lane centre line
    text += (f'DEF WEBOTS_VEHICLE0 BmwX5 {{\n  translation {p[0]:.3f} {p[1]:.3f} 0.32\n'
             f'  rotation 0 0 1 {yaw:.5f}\n  name "vehicle"\n  controller "<extern>"\n}}\n')
    print(f'ego car on lane {a.start_lane} at ({p[0]:.1f}, {p[1]:.1f}), heading {math.degrees(yaw):.0f} deg')

    if a.hospital:
        m = re.search(rf'SimpleBuilding \{{\n  translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)\n(?:  [^\n]*\n)*?  name "{re.escape(a.hospital)}"',
                      text)
        if not m:
            raise SystemExit(f'building {a.hospital!r} not found')
        hx, hy = float(m.group(1)), float(m.group(2))
        block = text[m.start():text.index('\n}', m.start())]
        floors = int(re.search(r'floorNumber (\d+)', block).group(1)) if 'floorNumber' in block else 3
        fh = float(re.search(r'floorHeight ([\d.]+)', block).group(1)) if 'floorHeight' in block else 3.0
        top = floors * fh + 0.6
        text += (f'DEF HOSPITAL_SIGN Pose {{\n  translation {hx} {hy} {top:.2f}\n  children [\n'
                 '    Shape { appearance PBRAppearance { baseColor 0.85 0.05 0.05 roughness 0.6 metalness 0 }'
                 ' geometry Box { size 7 2.2 1.2 } }\n'
                 '    Shape { appearance PBRAppearance { baseColor 0.85 0.05 0.05 roughness 0.6 metalness 0 }'
                 ' geometry Box { size 2.2 7 1.2 } }\n  ]\n}\n')
        print(f'hospital: {a.hospital} at ({hx:.1f}, {hy:.1f}), red cross on the roof at {top:.1f} m')

    out = os.path.join(wdir, f'{a.out}.wbt')
    open(out, 'w').write(text)
    subprocess.check_call([sys.executable, os.path.join(HERE, 'install_sensors.py'), out, '--def', 'WEBOTS_VEHICLE0'])
    subprocess.check_call([sys.executable, os.path.join(HERE, 'extract_signals.py'), out])
    subprocess.check_call([sys.executable, os.path.join(HERE, 'extract_pois.py'), out] +
                          (['--hospital', a.hospital] if a.hospital else []))
    print(f'-> {out}')


if __name__ == '__main__':
    main()
