from renderer import Camera, PointLight
from shapes import Sphere

from copy import deepcopy
import yaml
from yaml import CLoader as Loader

def yaml_file_to_world_objects(file_path):
    tree = None
    with open(file_path, 'r') as f:
        tree = yaml.load(f, Loader=Loader)

    if tree is None:
        return []

    rv = {'camera':None,
          'lights':[],
          'world':[]}

    defines = {}
    extends_map = {}

    for obj in tree:
        if "define" in obj:
            k = obj["define"]
            v = obj.get("value")
            opt = obj.get("extend")
            defines[k] = v
            if opt is not None:
                extends_map[k] = opt

    # replace 'extends' in defines map
    for obj_name in extends_map:
        parent_name = extends_map[obj_name] # name of object which will be extended
        parent_value = defines[parent_name]
        child_value = defines[obj_name] # name of object with 'extends' keyword
        new_parent_value = deepcopy(parent_value)
        if type(new_parent_value) == dict:
            # assume child value is same type
            for k in child_value:
                new_parent_value[k] = child_value[k]
            defines[obj_name] = new_parent_value

    # replace occurrences of previous defines in the tree
    for obj in tree:
        for k in defines:
            if "value" in obj and k in obj["value"]:
                new_parent_value = deepcopy(defines[k])
                i = obj["value"].index(k)
                del(obj["value"][i])
                for item in new_parent_value:
                    obj["value"].insert(i, item)
                    i += 1
            if "material" in obj and k in obj["material"]:
                if type(obj["material"]) == str:
                    obj["material"] = deepcopy(defines[k])
                elif type(obj["material"]) == dict:
                    tmp = obj["material"]
                    obj["material"] = deepcopy(defines[k])
                    for j in tmp:
                        obj["material"][j] = tmp[j]
            if "transform" in obj and k in obj["transform"]:
                i = obj["transform"].index(k)
                del(obj["transform"][i])
                for item in deepcopy(defines[k]):
                    obj["transform"].insert(i, item)
                    i += 1
            if "extend" in obj and k in obj["extend"]:
                del(obj["extend"])
                if "value" in obj and k in obj["value"]:
                    obj["value"][k] = deepcopy(defines[k])
                elif "value" in obj:
                    if type(obj["value"]) == dict:
                        tmp = obj["value"]
                        obj["value"] = deepcopy(defines[k])
                        for j in tmp:
                            obj["value"][j] = tmp[j]
                    else:
                        for item in deepcopy(defines[k]):
                            obj["value"].insert(0, item)
                else:
                    obj["value"] = deepcopy(defines[k])

    # TODO handle plane and cube and i should be able to render cover.yml
    # TODO handle groups and I should be able to handle group.yml. This will require changes to the above code, I think
    for obj in tree:
        print(obj)
        if "add" in obj:
            if obj["add"] == "camera":
                rv['camera'] = Camera.from_yaml(obj)
            elif obj["add"] == "light":
                rv['lights'].append(PointLight.from_yaml(obj))
            elif obj["add"] == "sphere":
                rv['world'].append(Sphere.from_yaml(obj))
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
    return rv

if __name__ == '__main__':
    x = yaml_file_to_world_objects("cover.yml")
    print(x)