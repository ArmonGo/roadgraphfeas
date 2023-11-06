import overpass
from shapely import ops
from shapely.geometry import LineString, MultiLineString, Point


class OverpassRoadLoader:
    def __init__(self, transformer=None, types=None):
        self.transformer = transformer
        if types is None:
            types = [
                "motorway",
                "trunk",
                "primary",
                "secondary",
                "tertiary",
                "unclassified",
                "residential",
                "motorway_link",
                "trunk_link",
                "primary_link",
                "secondary_link",
                "tertiary_link",
                "living_street"
                
            ]
        self.filter = f'[highway~"^({"|".join(types)})$"]'
        self.verbosity = "geom"

    def _process(self, lines):
        lines_processed = []
        for line in lines:
            line_processed = [
                Point(
                    self.transformer.transform(point[0], point[1])
                    if self.transformer
                    else (point[0], point[1])
                )
                for point in line
            ]
            lines_processed.append(LineString(line_processed))
        mls = MultiLineString(lines_processed)
        mls = ops.transform(lambda *args: args[:2], mls)
        return mls

    def load(self, query):
        roads_processed = {}
        try:
            api = overpass.API(timeout=600)
            response = api.get(query, verbosity=self.verbosity)
        except:
            print("query", query)
        
        for way in response.features:
            if way["geometry"]["type"] != "LineString":
                continue
            roads_processed[way.id] = way["geometry"]["coordinates"]
        return self._process(roads_processed.values())

    def load_bbox(self, bbox):
        return self.load(f"way({bbox[1]},{bbox[0]},{bbox[3]},{bbox[2]}){self.filter};")

    def load_radius(self, radius, lon, lat):
        return self.load(f"way(around:{radius},{lat},{lon}){self.filter};")
