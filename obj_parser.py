from vector import point
from shapes import Material, Group, Triangle
from matrix import matrix4x4identity

class OBJParser(object):
    def __init__(self, open_file, default_material=Material()):
        self.open_file = open_file
        self.default_material = default_material
        self.vertices = []
        self.vertices.append(None)
        self.named_groups = {'##default_group' : Group(self.default_material, matrix4x4identity(), set())}
        self.current_group_name = '##default_group'
        self.default_group = self.named_groups['##default_group']
        self._parse_file()

    def _parse_file(self):
        for line in self.open_file:
            _line = line.strip()
            if len(_line) > 0:
                self._parse_line(_line)

    def _parse_line(self, line):
        lsp = line.split(' ')
        if '' in lsp:
            lsp.remove('')
        if lsp[0] == 'v':
            self.vertices.append(self._parse_vertex(lsp))
        elif lsp[0] == 'f':
            self.named_groups[self.current_group_name].add_child(self._fan_triangulation(lsp))
        elif lsp[0] == 'g':
            self._parse_group(lsp)

    def _parse_group(self, line):
        if line[1] not in self.named_groups:
            self.named_groups[line[1]] = Group(self.default_material, matrix4x4identity(), set())
        self.current_group_name = line[1]

    def _parse_vertex(self, line):
        try:
            return point(float(line[1]), float(line[2]), float(line[3]))
        except ValueError as e:
            print(e, line)
            raise

    def _fan_triangulation(self, line):
        triangles = []
        if '/' in line[1]:
            line[1] = line[1].split('/')[0]
        if '/' in line[2]:
            line[2] = line[2].split('/')[0]
        for i in range(2, len(line) - 1):
            if '/' in line[i+1]:
                line[i+1] = line[i+1].split('/')[0]
            triangles.append(Triangle(Material(), matrix4x4identity(),
                           self.vertices[int(line[1])],
                           self.vertices[int(line[i])],
                           self.vertices[int(line[i+1])]))
        return triangles


def parse_obj_file(open_file):
    parser = OBJParser(open_file)
    print('vertices', parser.vertices)
    for group in parser.named_groups:
        print(group, parser.named_groups[group])

def obj_to_group(parser):
    obj_group = Group(Material(), matrix4x4identity(), set())
    obj_group.add_child([v for k,v in parser.named_groups.items()])
    return obj_group

def obj_file_to_group(file_path, material=Material()):
    with open(file_path, 'r') as f:
        parser = OBJParser(f, default_material=material)
        return obj_to_group(parser)

if __name__ == '__main__':
    test_obj_file = """
    v -1 1 0
    v -1.0000 0.0000 0.0000
    v 1 0 0
    v 1 1 0
    v 0 2 0

    f 1 2 3 4 5

    g FirstGroup
    f 1 2 3
    g SecondGroup
    f 1 3 4
    """
    parse_obj_file(test_obj_file.split('\n'))
