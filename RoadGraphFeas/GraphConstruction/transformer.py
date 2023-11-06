import pyproj
import osmnx as ox

def get_local_crs(x,y,radius):  
    trans = ox.utils_geo.bbox_from_point((y, x), dist = radius, project_utm = True, return_crs = True)
    return trans[-1]

class CoordinateTransformer:
    def __init__(
        self,
        from_crs,
        centroid,
        radius, 
        reverse_before=False,
        reverse_after=False,
        invert_x_before=False,
        invert_y_before=False,
        invert_x_after=False,
        invert_y_after=False,
    ):
        
        self.from_crs = from_crs
        self.to_crs = pyproj.CRS(get_local_crs(centroid[0], centroid[1],radius)) # based on centriod and radius generate the crs file no hard code anymore
        self.reverse_before = reverse_before
        self.reverse_after = reverse_after
        self.invert_x_before = invert_x_before
        self.invert_y_before = invert_y_before
        self.invert_x_after = invert_x_after
        self.invert_y_after = invert_y_after
        assert (
            self.to_crs.coordinate_system.name == "cartesian"
        ), self.to_crs.coordinate_system.name
        self.transformer = pyproj.Transformer.from_crs(
            self.from_crs, self.to_crs, always_xy=True
        )

    def transform(self, lon, lat):
        if self.reverse_before:
            lon, lat = lat, lon
        if self.invert_x_before:
            lon = -lon
        if self.invert_y_before:
            lat = -lat
        x, y = self.transformer.transform(lon, lat)
        tx, ty = (x, y) if not self.reverse_after else (y, x)
        if self.invert_x_after:
            tx = -tx
        if self.invert_y_after:
            ty = -ty
        return tx, ty
