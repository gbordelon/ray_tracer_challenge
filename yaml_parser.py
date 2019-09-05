import json
import yaml
from yaml import CLoader as Loader

def yaml_file_to_world_objects(file_path):
    tree = None
    with open(file_path, 'r') as f:
        tree = yaml.load(f, Loader=Loader)

    if tree is None:
        return []

    rv = []
    # switch on "add" keyed objects
    for obj in tree:
        if "add" in obj:
            if obj["add"] == "camera":
                rv.append(parse_camera(obj))
            elif obj["add"] == "light":
                pass
            elif obj["add"] == "sphere":
                pass
            elif obj["add"] == "plane":
                pass
            elif obj["add"] == "cube":
                pass
            elif obj["add"] == "cone":
                pass
            elif obj["add"] == "cylinder":
                pass
            elif obj["add"] == "group":
                pass

def parse_camera(node):
    pass

if __name__ == '__main__':
    x = yaml_file_to_world_objects("cover.yml")
    print(x)