###### AN AUTOMATED MESHING TOOL FOR FEM SIMULATIONS ######
# 
# ACCEPTS VTK surfaces and parses them with OpenCascadeCAD/gmsh to create completed 3D meshes
# for use with MOOSE or other FEM frameworks
import pyvista as pv
import math
import gmsh
import numpy as np
import pandas as pd
from collections import defaultdict

''' Surface name and element accounting '''
# Dimention and ID tracker, takes in surface gmsh element tags e.g. (dimention, ID's) tuples 
# and their names and tracks changes passed from fragmentation operations or other boolean mesh operations

class DimensionIDTracker:

    ''' Initializes dictionary'''
    def __init__(self):
        self.data = {}  # Stores {(dim, id): name}
        self.parent_child_map = defaultdict(list)  # Stores parent-child relationships
        self.debug_flag=True
    
    def add_entry(self, dim_ids, names):
        """
        Adds multiple entries to the tracker.
        :param dim_ids: List of tuples [(dimension, id), ...]
        :param names: List of strings [name, ...]
        """
        if len(dim_ids) != len(names):
            raise ValueError("Dimension ID list and names list must be of the same length.")
        
        for dim_id, name in zip(dim_ids, names):
            if dim_id[0] not in {1, 2, 3}:
                raise ValueError("Dimension must be 1, 2, or 3")
            print("Adding ", name, "to ", dim_id)
            self.data[dim_id] = name
    
    def update_entries(self, parent_entries, child_entries):
        """
        Updates the tracker based on transformations.
        :param parent_entries: List of tuples [(orig_dim, orig_id), ...]
        :param child_entries: List of lists of tuples [[(new_dim. new_id)],[(new_dim, new_id), ...]...]
        Parent and child entries must be mapped one old to one new entry
        """
        if len(parent_entries) != len(child_entries): 
            raise ValueError("Original entries and new entries lists must be of the same length.")
        
        
        for orig, children in zip(parent_entries, child_entries):
            if self.debug_flag:
                print("Updating Orig: ",orig," with Children: ",children)
            if orig not in self.data:
                    raise KeyError(f"Original entry {orig} not found.")
            parent_name = self.data[orig]
            if self.debug_flag:
                print("Children: ",children)
            if children == []: # this conditional may be soon depreciated, only remains for the no fragenting corner case
                remove_name = "remove"
                self.data[orig] = remove_name   
            else:
                for new in children:
                    if self.debug_flag:
                        print(f"Adding {new} : {parent_name}")
                    #parent_name = self.data[orig]
                    self.data[new] = parent_name  # Inherit the name
                    self.parent_child_map[orig].append(new)  # Track the split
    
    def get_hierarchy(self):
        """
        Returns the parent-child hierarchy as a dictionary.
        """
        return dict(self.parent_child_map)
    
    def lookup_name(self, dim_id):
        """
        Looks up the name associated with a given dimension ID.
        :param dim_id: Tuple (dimension, id)
        :return: Name string if found, otherwise None
        """
        return self.data.get(dim_id, None)
    
    def get_name_counts(self, by_dimension=False):
        """
        Returns a count of each unique name in the dictionary.
        If by_dimension is True, returns a nested dictionary with counts per dimension.
        """
        if by_dimension:
            name_counts = defaultdict(lambda: defaultdict(int))
            print(self.data.items())
            for (dim, _), name in self.data.items():
                name_counts[dim][name] += 1
            return {dim: dict(names) for dim, names in name_counts.items()}
        else:
            name_counts = defaultdict(int)
            for name in self.data.values():
                name_counts[name] += 1
            return dict(name_counts)
    
    def get_sorted_by_prefix(self):
        """
        Returns a list of lists where elements are grouped and sorted by name prefix.
        """
        prefix_groups = defaultdict(list)
        
        for dim_id, name in self.data.items():
            prefix = name.split()[0] if " " in name else name  # Extract prefix
            prefix_groups[prefix].append((dim_id, name))
        
        return [sorted(group, key=lambda x: x[1]) for group in prefix_groups.values()]
    
    def __repr__(self):
        return f"Data: {self.data}\nParent-Child Map: {dict(self.parent_child_map)}"

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def load_surface(path):
    """Load STL or VTK into numpy arrays (points, triangles)."""
    mesh = pv.read(path)
    mesh.clean(inplace=True)
    points = mesh.points.copy()
    faces = mesh.faces.reshape(-1, 4)[:, 1:].copy()  # remove leading 3
    return points, faces


def boundary_edges(tris):
    """Return all edges that appear only once → boundary."""
    edges = {}
    for tri in tris:
        for i, j in [(0,1),(1,2),(2,0)]:
            e = tuple(sorted((tri[i], tri[j])))
            edges[e] = edges.get(e, 0) + 1
    return [e for e, c in edges.items() if c == 1]


def boundary_order(points, boundary_edges):
    """Order boundary edges into a continuous loop."""
    # adjacency
    adj = {}
    for i, j in boundary_edges:
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)

    # find start
    start = boundary_edges[0][0]
    loop = [start]
    prev = None
    cur = start

    while True:
        nxt = [x for x in adj[cur] if x != prev]
        if len(nxt)==0:
            break
        nxt = nxt[0]
        if nxt == start:
            break
        loop.append(nxt)
        prev, cur = cur, nxt

    return np.array(loop)


# -------------------------------------------------------
# Radial XY Extension (fixed)
# -------------------------------------------------------

def extend_radial(points, boundary_ids, bbox):
    """
    Extend *only boundary vertices* radially in XY until hitting bounding box.
    bbox = (xmin, xmax, ymin, ymax)
    """
    xmin, xmax, ymin, ymax = bbox

    P = points.copy()
    B = points[boundary_ids, :]

    # centroid for radial directions
    c = B[:, :2].mean(axis=0)

    new_boundary = []

    for p in B:
        x0, y0 = p[:2]
        dx, dy = x0 - c[0], y0 - c[1]

        if dx == 0 and dy == 0:
            dx = 1e-6

        tx = []
        if dx > 0: tx.append((xmax - x0)/dx)
        if dx < 0: tx.append((xmin - x0)/dx)
        if dy > 0: tx.append((ymax - y0)/dy)
        if dy < 0: tx.append((ymin - y0)/dy)

        t = min([tt for tt in tx if tt > 0])
        newx = x0 + t*dx
        newy = y0 + t*dy

        new_boundary.append([newx, newy, p[2]])

    new_boundary = np.array(new_boundary)
    P[boundary_ids] = new_boundary
    return P


# -------------------------------------------------------
# Convex Hull Extension
# -------------------------------------------------------

import numpy as np
from shapely.geometry import Polygon, Point, LineString

def extend_convex_hull(points, boundary_ids, bbox, scale=1.2):
    """
    Project boundary vertices radially to a scaled convex hull.
    """
    xy = points[:, :2]
    bpts = xy[boundary_ids]

    # Construct convex hull polygon
    hull = Polygon(bpts).convex_hull
    cx, cy = hull.centroid.xy
    cx, cy = float(cx[0]), float(cy[0])

    # scale hull outward
    scaled_coords = []
    for (x, y) in hull.exterior.coords:
        vx, vy = x - cx, y - cy
        scaled_coords.append((cx + scale * vx, cy + scale * vy))

    scaled_hull = Polygon(scaled_coords)
    hull_boundary = LineString(scaled_hull.exterior.coords)

    new_xy = xy.copy()

    for pid in boundary_ids:
        x, y = xy[pid]
        ray = LineString([(cx, cy), (x, y)])
        inter = ray.intersection(hull_boundary)

        if not inter.is_empty:
            # always take point, even if MultiPoint
            if hasattr(inter, "geoms"):
                inter = inter.geoms[0]
            new_xy[pid] = inter.x, inter.y

    P = points.copy()
    P[:, :2] = new_xy
    return P



# -------------------------------------------------------
# TPS Minimal Curvature Extension
# -------------------------------------------------------

def extend_tps(points, boundary_ids, bbox, grid_spacing, scale=1.5):
    xy = points[:, :2]
    bpts = xy[boundary_ids]

    # source: original boundary
    src = bpts

    # compute enlarged bbox
    xmin, xmax, ymin, ymax = bbox
    cx, cy = xy.mean(axis=0)

    width = (xmax - xmin) * scale
    height = (ymax - ymin) * scale

    # make target boundary ring
    xs = np.linspace(cx - width/2, cx + width/2, int(width / grid_spacing))
    ys = np.linspace(cx - height/2, cy + height/2, int(height / grid_spacing))

    xx, yy = np.meshgrid(xs, ys)
    dst = np.vstack([xx.ravel(), yy.ravel()]).T

    # TPS warp
    tps = Rbf(src[:,0], src[:,1], src[:,0], function='thin_plate')
    fx = tps(dst[:,0], dst[:,1])

    tps = Rbf(src[:,0], src[:,1], src[:,1], function='thin_plate')
    fy = tps(dst[:,0], dst[:,1])

    # map boundary pts to new location
    P = points.copy()
    P[boundary_ids, 0] = fx[:len(boundary_ids)]
    P[boundary_ids, 1] = fy[:len(boundary_ids)]
    return P


# -------------------------------------------------------
# Public API
# -------------------------------------------------------

def extend_surface(path, method="radial", bbox=None, grid_spacing=50, scale=1.0 ,output=None):
    """
    Extend surface edges while preserving original geometry.
    method = "radial" | "hull" | "tps"
    """
    points, tris = load_surface(path)

    # boundary detection
    b_edges = boundary_edges(tris)
    b_ids = boundary_order(points, b_edges)

    # default bounding box
    if bbox is None:
        xmin, xmax = points[:,0].min(), points[:,0].max()
        ymin, ymax = points[:,1].min(), points[:,1].max()
        pad = 0.1 * max(xmax-xmin, ymax-ymin)
        bbox = (xmin-pad, xmax+pad, ymin-pad, ymax+pad)

    # ---- apply extension ----
    if method == "radial":
        new_points = extend_radial(points, b_ids, bbox)
    elif method == "hull":
        new_points = extend_convex_hull(points, b_ids, bbox, scale)
    elif method == "tps":
        new_points = extend_tps(points, b_ids, bbox, grid_spacing, scale)
    else:
        raise ValueError("Unknown method")

    # save result
    mesh = pv.PolyData(new_points, np.hstack([np.full((len(tris),1),3), tris]))
    if output:
        mesh.save(output)

    return new_points, tris


""" Worker functions NEED TO IMPLEMENT INTO A WOKRING CLASS """

def create_rotated_box(xmin, xmax, ymin, ymax, zmin, zmax, rot_center='centroid', azimuth_deg=0.0, tag=-1):
    """
    Create a rectilinear (box) volume in Gmsh that can be rotated about its center
    by a specified azimuth angle (in degrees).

    Parameters
    ----------
    xmin, xmax : float
        X-coordinate limits of the box.
    ymin, ymax : float
        Y-coordinate limits of the box.
    zmin, zmax : float
        Z-coordinate limits of the box.
    azimuth_deg : float, optional
        Rotation angle in degrees about the box center (around Z-axis).
        Default is 0.0 (no rotation).
    tag : int, optional
        Optional tag for the created volume.
    
    Returns
    -------
    vol_tag : int
        Tag of the created volume.
    center : tuple[float, float, float]
        Coordinates of the box center.

    example usage:
        box_tag, center = create_rotated_box(-5, 0, -5, 5, 10, 5, azimuth_deg=-360, tag=-1)
    """

    if rot_center=='centroid':
        # Compute box center
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

    if rot_center=='origin':
        cx=0.
        cy=0.
        cz=0.
    
    if type(rot_center) == tuple:
        cx=rot_center[0]
        cy=rot_center[1]
        cz=rot_center[2]

    # Create box volume
    vol_tag = gmsh.model.occ.addBox(xmin, ymin, zmin,
                                    (xmax - xmin), (ymax - ymin), (zmax - zmin), tag)

    # Apply rotation about center if azimuth is non-zero
    if abs(azimuth_deg) > 1e-12:
        theta = math.radians(azimuth_deg) 
        # rotation around Z-axis
        gmsh.model.occ.rotate(
            [(3, vol_tag)],  # 3 = volume dimension
            cx, cy, cz,      # center of rotation
            0, 0, 1,         # rotation axis (Z)
            theta
        )
        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
    return vol_tag, (cx, cy, cz)


def clip_polydata(polydata, xmin, xmax, ymin, ymax, zmin, zmax):
    # Create a box filter for clipping input
    bounds = [xmin, xmax, ymin, ymax, zmin, zmax]
    clipped = polydata.clip_box(bounds=bounds, invert=False)
    return unstructured_to_exact_polydata(clipped)

def filter_tuples_by_first_entry(tuples_list, value):
    return [tup for tup in tuples_list if tup[0] == value]

def make_LOG_entry(meshLOG, meshLogCols, meshLogData):
        """
        Add an entry to a pandas DataFrame

        Parameters:
        meshLOG (pandas DataFrame): input DataFrame that will added to
        meshLogCols [list]: Columns to add data to 
        meshLogData [list]: Data to add to columns
            ***** both meshLogData and ...Cols must be collated ***** 
        """

        #logEntry = res = dict(map(lambda i,j : (i,j) , meshLogCols , meshLogData))
        #meshLOG = pd.DataFrame(columns=meshLogCols)
        #meshLOG.jo(pd.DataFrame(logEntry))
        colEntry = []
        #colEntry.append(meshLogCols)
        colEntry.append(meshLogData)
        meshLOG = pd.concat((meshLOG, pd.DataFrame(colEntry, columns=meshLogCols)))
        outputmeshLOG = pd.DataFrame(colEntry, columns=meshLogCols)
        print(outputmeshLOG)
        return meshLOG

def unique_points(list1, list2):
    """
    Returns a list of points from the first list (list1) that are not present in the second list (list2).

    Parameters:
    list1 (list of lists): The first list of [x, y, z] data points.
    list2 (list of lists): The second list of [x, y, z] data points to exclude from list1.

    Returns:
    list of lists: A list containing only the points from list1 that are not in list2.
    """
    
    # Convert list of lists to set of tuples for both lists
    set1 = set(map(tuple, list1))
    set2 = set(map(tuple, list2))
    
    # Find the difference between set1 and set2
    unique_points_set = set1 - set2
    
    # Convert the resulting set back to a list of lists
    #unique_points_list = list(map(list, unique_points_set))
    unique_points_list = list(map(tuple, unique_points_set))
    
    # Optional: Sort the list of unique points for consistency
    unique_points_list.sort()
    
    return unique_points_list


def unstructured_to_exact_polydata(unstructured_grid):
    """
    Converts a pyvista.UnstructuredGrid surface to a pyvista.PolyData surface
    with the exact same points and lines from the input surface.
    
    Parameters:
    - unstructured_grid (pv.UnstructuredGrid): The input unstructured grid.
    
    Returns:
    - pv.PolyData: The output polydata surface with the same points and lines.
    """
    if not isinstance(unstructured_grid, pv.UnstructuredGrid):
        raise TypeError("Input must be a pyvista.UnstructuredGrid object.")
    
    # Extract points and cells directly
    points = unstructured_grid.points
    cells = unstructured_grid.cells

    # Create PolyData from points and cells
    polydata_surface = pv.PolyData(points, cells)
    
    return polydata_surface

def add_intersection_points(surfaces):
    """
    Add intersection points to each PyVista polydata surface based on intersections
    with other surfaces.

    Parameters:
        surfaces (list of pv.PolyData): List of PyVista PolyData surfaces.

    Returns:
        list of pv.PolyData: List of modified PolyData surfaces with added points.
    """
    if not all(isinstance(surface, pv.PolyData) for surface in surfaces):
        raise ValueError("All inputs must be PyVista PolyData surfaces.")

    modified_surfaces = []

    for i, surface_a in enumerate(surfaces):
        # Create a copy of the current surface to avoid modifying the input
        modified_surface = surface_a.copy()
        print(type(surface_a))
        #time.sleep(2)
        for j, surface_b in enumerate(surfaces):
            if i != j:  # Avoid self-intersection
                # Compute the intersection between surface_a and surface_b
                print(type(surface_b))
                intersection, s1, s2 = surface_a.intersection(surface_b)
                print(type(intersection))
                #time.sleep(2)
                # Check if the intersection has points
                print(intersection.n_points)
                #time.sleep(2)
                if intersection and intersection.n_points > 0:
                    # Add intersection points to the modified surface
                    points_to_add = pv.PolyData(intersection.points)
                    modified_surface = modified_surface.merge(points_to_add)

        modified_surfaces.append(modified_surface.clean(inplace=True).delaunay_2d())

    return modified_surfaces

""" Function to write out the boundary mesh from the gmsh fragment operation, the combination of all these meshs are the dividing (fault) surface
     this will help produce the mesh block for the mooose input"""

def surf_list(surface_name="Surface_", start=1, end=2):
    boundary_list = ' '.join([f'{surface_name}{i:04d}' for i in range(start, end)])
    #print(boundary_list)
    return boundary_list


def sort_by_prefix(prefixes, strings):
    """
    Groups strings into lists based on their prefixes.

    Parameters:
    prefixes (list of str): List of prefixes.
    strings (list of str): List of strings to be sorted.

    Returns:
    list of lists: A list where each sublist contains strings matching a prefix.
    """
    grouped = defaultdict(list)

    for s in strings:
        for prefix in prefixes:
            if s.startswith(prefix):
                grouped[prefix].append(s)
                break  # Stop checking once a match is found

    return [grouped[prefix] for prefix in prefixes]

def rotated_box_face_directions(azimuth_deg):
    """
    Given an azimuth in degrees (clockwise from North),
    return the direction names for the six box faces after rotation.

    The base orientation (azimuth = 0) is:
        1: West
        2: East
        3: South
        4: North
        5: Base
        6: Top

    For rotations, faces 1–4 rotate about the vertical axis (Z),
    while Base and Top remain fixed.

    Parameters
    ----------
    azimuth_deg : float
        Rotation angle in degrees (clockwise from North).

    Returns
    -------
    list[str]
        List of 6 strings corresponding to face directions after rotation.
    """

    # Normalize azimuth between 0–360
    az = azimuth_deg % 360.0

    # Define 16 compass sectors (each 22.5° wide)
    # The midpoints define where each label changes
    compass_labels = [
        ("North", 0),
        ("NNE", -22.5),
        ("NE", -45),
        ("ENE", -67.5),
        ("East", -90),
        ("ESE", -112.5),
        ("SE", -135),
        ("SSE", -157.5),
        ("South", -180),
        ("SSW", -202.5),
        ("SW", -225),
        ("WSW", -247.5),
        ("West", -270),
        ("WNW", -292.5),
        ("NW", -315),
        ("NNW", -337.5),
    ]

    def az_to_dir(angle):
        """Map azimuth to nearest compass label."""
        # Wrap 360 back to 0
        angle = angle % 360
        for i in range(len(compass_labels)):
            name, center = compass_labels[i]
            next_center = compass_labels[(i + 1) % len(compass_labels)][1]
            # Compute sector midpoint range
            lower = center - 11.25
            upper = center + 11.25
            # Handle wrap around
            if lower < 0:
                if angle >= (360 + lower) or angle < upper:
                    return name
            elif upper >= 360:
                if angle >= lower or angle < (upper - 360):
                    return name
            elif lower <= angle < upper:
                return name
        # Fallback (should not happen)
        return "North"

    # Base face azimuths (before rotation)
    # West = 270°, East = 90°, South = 180°, North = 0°
    base_azimuths = {
        1: 90,
        2: 270,
        3: 180,
        4: 0,
    }

    # Apply rotation (clockwise, so add azimuth)
    rotated_dirs = {}
    for face, base_az in base_azimuths.items():
        new_az = (base_az + az) % 360
        rotated_dirs[face] = az_to_dir(new_az)

    # Base and Top don’t change
    rotated_dirs[5] = "Base"
    rotated_dirs[6] = "Top"

    # Return in 1–6 order
    return [rotated_dirs[i] for i in range(1, 7)]



def export_divided_gmsh_volume_Compound_surface_accounting(
    surfaces, 
    box_dimensions, 
    output_filename="output.msh",
    surface_names={},
    field_size=500
    
):
    """
    Divides a rectangular box using a series of PyVista surfaces and exports a Gmsh volume mesh.
    Groups and labels surfaces created after each division.

    Parameters:
    surfaces (list): List of PyVista PolyData surfaces to divide the box.
    box_dimensions (tuple): The dimensions of the rectangular box (length, width, height).
    output_filename (str): The filename for the output .msh file.
    """

    gmsh.initialize()
    gmsh.model.add("divided_volume")

    # Define a rectangular box in Gmsh using OCC kernel
    #length, width, height = box_dimensions
    #box = gmsh.model.occ.add_box(0, 0, 0, length, width, height)
    xmin,xmax,ymin,ymax,zmin,zmax = box_dimensions
    box = gmsh.model.occ.add_box(xmin, ymin, zmin, xmax, ymax, zmax)
    # Synchronize after adding the box
    gmsh.model.occ.synchronize()
    
    #gmsh.model.mesh.setSize(gmsh.model.getBoundary([(3, box)]), 200)

    # Alternatively, you can set a field-based size using a background field
    # Field 1: Uniform mesh size over the whole domain
    #gmsh.model.mesh.field.add("Constant", 1)
    #gmsh.model.mesh.field.setNumber(1, "VIn", 200)  # uniform element size
    #gmsh.model.mesh.field.setAsBackgroundMesh(1)
  
    
    # Convert PyVista surfaces to Gmsh geometries
    gmsh_surfaces = []
    for i, surface in enumerate(surfaces):
        points = surface.points
        faces = surface.faces.reshape((-1, 4))[:, 1:]  # Assuming triangular faces

        # Add points to Gmsh
        gmsh_points = []
        for pt in points:
            gmsh_points.append(gmsh.model.occ.add_point(pt[0], pt[1], pt[2]))

        # Add faces as Gmsh surfaces
        gmsh_curves = []
        for face in faces:
            lines = []
            for j in range(len(face)):
                p1 = gmsh_points[face[j]]
                p2 = gmsh_points[face[(j + 1) % len(face)]]
                lines.append(gmsh.model.occ.add_line(p1, p2))
            loop = gmsh.model.occ.add_curve_loop(lines)
            surface_tag = gmsh.model.occ.add_plane_surface([loop])
            gmsh_curves.append(surface_tag)

        # Store the surface tags for later use
        gmsh_surfaces.append(gmsh_curves)
    fragment_ov = []
    fragment_ovv = []
    # Synchronize OCC geometry definitions
    gmsh.model.occ.synchronize()
    all_fragmenting_surfaces = []
    all_fragmenting_surface_names = []
    surface_physical_names = []
    compound_surface_id_idx = []
    # Use the surfaces to divide the rectangular box into different zones
    partitions = [(3, box)]  # The box is initially the entire volume
    for surf_idx in range(len(surface_names)):
        print("Surf idx: " + str(surf_idx))
        SurfName = surface_names[surf_idx]
        # Fragment the current partitions using the new surface
        ovv, ov = gmsh.model.occ.fragment(partitions, [(2, surface_tag) for surface_tag in gmsh_surfaces[surf_idx]], removeTool=True, removeObject=True)
        gmsh.model.occ.synchronize()

        print(SurfName)
        dim2_surface_tags = filter_tuples_by_first_entry(ovv, 2)   # this function only returns the 2D surface tags
        frag_surf_tags = [tags[1] for tags in dim2_surface_tags]
        # Assign physical groups to the new surfaces created by this fragmentation
        all_fragmenting_surface_names = []  # holder of the Physical group names to output later
        for idx, surf in enumerate(frag_surf_tags, start=1):
            
            group_tag = 1000 * (surf_idx + 1) + int(surf)  # Create a unique identifier for the surface group (six digit is surfaces and four digit is partition)
            gmsh.model.add_physical_group(2, [surf], tag=group_tag)
            gmsh.model.set_physical_name(2, group_tag, f"{SurfName}_{idx:04d}")
            all_fragmenting_surface_names.append(f"{SurfName}_{idx:04d}") # append the name to the name list, will be added later to the surface_physical_names list
            if idx == 1:
                compound_surface_id_idx.append(gmsh.model.getEntitiesForPhysicalName(f"{SurfName}_{idx:04d}")) # this is a strage OCC behavior, the compound surface is indexed 100000 plus this first new entity, not exactly sure why
            # Get updated partitions
            partitions = gmsh.model.occ.get_entities(dim=3)

            gmsh.model.occ.synchronize()
        surface_physical_names.append(all_fragmenting_surface_names) # These lists are coded in order by the surface names list input (the surfaces are in the names too so it should be obvious) 
        
        
        # create a compound surface from the fragmented surface tags, use the <**surface group**000> tag (e.g. 1000, 2000, etc) for the entity id (compounding surface entities start numbering at 1)     
        #gmsh.model.mesh.setCompound(2, frag_surf_tags)
        #surf_group_tag = 1000 * (surf_idx + 1)
        #gmsh.model.add_physical_group(2, [compound_surf], tag=surf_group_tag)
        #gmsh.model.set_physical_name(2, surf_group_tag, f"{SurfName}")



        #gmsh.model.mesh.setCompound(2, frag_surf_tags)

        ''' Moving this down to after the generate command
        print(compound_surface_id_idx)
        compound_tag = 100000 + compound_surface_id_idx[0][1]
        print("compound surf tag= " + str(compound_tag))
        new_tag = (100000 * (surf_idx+1))
        gmsh.model.add_physical_group(2, [compound_tag], tag=new_tag)
        gmsh.model.set_physical_name(2, new_tag, SurfName)
        #gmsh.model.setEntityName(2, compound_tag, name=f"{SurfName}")
        '''


        fragment_ov.append(ov)
        fragment_ovv.append(ovv)
        all_fragmenting_surfaces.extend(ovv)
    # Collect surfaces that are not on fragmenting the volume, these should be exterior 
    #all_surfaces = [tags[1] for tags in gmsh.model.get_entities(2)]
    gmsh.model.mesh.reclassifyNodes()
    all_surfaces = gmsh.model.get_entities(2)   # Experimental
    exterior_surfaces = unique_points(all_surfaces, all_fragmenting_surfaces)

    volume_names = []
    exterior_surface_names = []
    for idx, (dim, surf) in enumerate(exterior_surfaces, start=1):
        print("ext surfaces " + str(surf))
        group_tag = 10000 + int(surf)  # Create a unique identifier for the surface group (six digit is surfaces and four digit is partition)
        gmsh.model.add_physical_group(2, [surf], tag=group_tag)
        gmsh.model.set_physical_name(2, group_tag, f"Ext_Surf_{idx:02d}")
        exterior_surface_names.append(f"Ext_Surf_{idx:02d}")

    # Assign unique identifiers to each zone (volume)
    for idx, (dim, volume) in enumerate(partitions, start=1):
        gmsh.model.add_physical_group(3, [volume], tag=idx)
        gmsh.model.set_physical_name(3, idx, f"Volume_{idx:02d}")
        volume_names.append(f"Volume_{idx:02d}")
    gmsh.model.occ.synchronize()
   
   
   
    # Generate the mesh
    pnt_entities = gmsh.model.occ.get_entities(dim=0)

    gmsh.model.mesh.setSize(pnt_entities, field_size) #field size determines the maximum distance between point node elements

    # Alternatively, you can set a field-based size using a background field
    # Field 1: Uniform mesh size over the whole domain
    #gmsh.model.mesh.field.add("Constant", 1)
    #gmsh.model.mesh.field.setNumber(1, "VIn", 200)  # uniform element size
    #gmsh.model.mesh.field.setAsBackgroundMesh(1)

    gmsh.model.mesh.generate(3)   # generate 3D mesh
    #for surf_idx in range(len(compound_surface_id_idx)):
    #    print(compound_surface_id_idx)
    #    compound_tag = 100000 + compound_surface_id_idx[surf_idx][0][1]
    #    print("compound surf tag= " + str(compound_tag))
    #    new_tag = (100000 * (surf_idx+1))
    #    gmsh.model.add_physical_group(2, [compound_tag], tag=new_tag)
    #    gmsh.model.set_physical_name(2, new_tag, surface_names[surf_idx])
        #gmsh.model.setEntityName(2, compound_tag, name=f"{SurfN
    # Export mesh to file
    gmsh.write(output_filename)
    gmsh.finalize()
    print(f"Mesh exported to {output_filename}")
    return fragment_ov, fragment_ovv, gmsh_surfaces, surface_physical_names, volume_names, exterior_surface_names

def build_mesh_from_surfaces(all_horizons=[], volume_names=[], vol_surface_names=[], all_faults=[], 
                             fault_surface_names=[], model_name="divided_volume", output_file='./mesher.msh', 
                             bounding_box=[0,1,0,1,0,1], fieldsize=500, rotation=0.0, Dim=3, rot_center='origin', debug=True, mesh=False):

    """
    Finite Element Method (FEM) mesh constructor from "watertight" polydata surfaces.

    Parameters:
    all_horizons [list of obj]: pyvista Polydata surfaces for dividing the mesh, from bottom to top of the mesh
    volume_names [list of str]: names of volumes, must be in corresponding order to all_horizon parameter with a leading name that indicates the bottom volume e.g 'basement'
    vol_surface_names [list of str.]: names of horizon surfaces, in all_surfaces

    all_faults [list of obj]: pyvista Polydata surfaces for dividing the mesh, these surfaces will not result in volume renaming
    fault_surface_names [list of str.]: names of fault surfaces, in all_faults

    model_name [str]: gmesh mesh name 
    output_file [str]: filepath for saving out the completed mesh

    bounding_box [list]: extents of box volume  [ xmin, xmax, ymin, ymax, zmin, zmax ]


    Outputs: dim2 and dim3 element list of lists of the surfaces sorted by their name prefix

    """ 

    xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
    extents = (xmin, (xmax-xmin), ymin, (ymax-ymin), zmin, (zmax-zmin))
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.model.add(model_name)

        # Define a rectangular box in Gmsh using OCC kernel
        #length, width, height = box_dimensions
        #box = gmsh.model.occ.add_box(0, 0, 0, length, width, height)

    #gmsh.model.occ.add_box(extents[0], extents[2], extents[4], extents[1], extents[3], extents[5])

    box_tags, center = create_rotated_box(xmin, xmax, ymin, ymax, zmin, zmax, rot_center=rot_center, azimuth_deg=rotation, tag=1)

    if debug:
        print("box tags ",box_tags, " center: ",center, "Rotation Azimuth Deg: ", rotation)
    #box = gmsh.model.occ.get_entities(dim=3)
        # Synchronize after adding the box
       
    #face_names =  ['West', 'East', 'South', 'North', 'Base', 'Top'] #box surfaces creation order (ZY- , ZY+ , ZX- , ZX+ , XY- , XY+) or (W, E, S, N, Dn, Up)
    face_names = rotated_box_face_directions(rotation)
    #collect all the surfaces naming components to sort dim2 element names after fragmenting
    all_dim2_prefix = vol_surface_names+fault_surface_names+face_names
    
    
    tracker = DimensionIDTracker()
    
    gmsh.model.occ.synchronize()

    box = gmsh.model.occ.get_entities(dim=3)


    box_2dim = gmsh.model.occ.get_entities(dim=2)
    if debug:
        print("Box Tag ", box, [f"box_2_dim tags: {tags}, " for tags in zip(box_2dim, face_names)])
    tracker.add_entry(box_2dim, face_names)
    
    

    """ First Volume naming """


    #print(box, volume_names[0])
    # Adding Box surface face names
    if debug:
        print("Tracker: Volume BOX", box[0], "Named: ", volume_names[0])
    tracker.add_entry([box[0]], [volume_names[0]]) 


    """_____________________________  STRATIGRAPHY FRAGMENTING LOOP __________________________________"""
    # FIRST WE CUT THE BLOCK FROM THE BOTTOM UP WITH STRATIGRAPHY (FAULTS ARE HANDLED IN THE NEXT LOOP)
    for surface, name, vol_name in zip(all_horizons, vol_surface_names, volume_names[1:]):
        current_vols = gmsh.model.occ.get_entities(dim=3)

        """Adding surfaces to OpenCascade"""

        points = surface.points
        faces = surface.faces.reshape((-1, 4))[:, 1:]
        #points.round(2), faces 

        gmsh_points = []
        for pt in points:
            gmsh_points.append(gmsh.model.occ.add_point(pt[0], pt[1], pt[2]))
        #print("Points ",gmsh_points)

        gmsh_curves = []
        for face in faces:
                lines = []
                for j in range(len(face)):
                    p1 = gmsh_points[face[j]]
                    p2 = gmsh_points[face[(j + 1) % len(face)]]
                    lines.append(gmsh.model.occ.add_line(p1, p2))
                    #print(f"line {lines} {j}")
                loop = gmsh.model.occ.add_curve_loop(lines)
                #print("loop ", loop)
                surface_tag = gmsh.model.occ.add_plane_surface([loop])
                #print("adding surface tag : ", surface_tag)
                gmsh_curves.append(surface_tag)
                gmsh.model.occ.synchronize()
        #print("Curves :", gmsh_curves)
        #gmsh.model.occ.removeAllDuplicates()  ##### TESTING - Added for PLC error #### 
        gmsh.model.occ.synchronize()
        #gmsh.model.occ.synchronize()
        
        
        namer = [name] * len(gmsh_curves)
        #print(namer)
        tracker.add_entry([(2, plane_parts) for plane_parts in gmsh_curves], namer)

        #gmsh.model.occ.synchronize()
    
        #all_2dim = gmsh.model.occ.get_entities(dim=2) #box creation order (ZY- , ZY+ , ZX- , ZX+ , XY- , XY+) or (W, E, S, N, Dn, Up)
        #print("Box 2 dim tags tuples: ", all_2dim)
        box_3dim = gmsh.model.occ.get_entities(dim=3) #original box
        

        #box_2dim = gmsh.model.occ.get_entities(dim=2)
        #boxtags = [tags[1] for tags in box_2dim]

        #surf_dict = "surf"

        #print("Box tags without dimension: ",boxtags)

        # return all dim2 surface elements from the current model
        all_surf = gmsh.model.occ.get_entities(dim=2)
    
        #print("gmsh _curves ",gmsh_curves)
        
        #combine all the elements in the volume, we need them all so that the fragment operation reports all the parent child relationships correctly without gaps
        box_surfaces = all_surf+box_3dim
        if debug:
            print(f"Cutting surface, {name} - All box surfaces/Vols: ", box_surfaces, "box_3dim: ", box_3dim)
        
        ovv, ov = gmsh.model.occ.fragment(box_surfaces, all_surf, removeTool=True, removeObject=True) #[(2, plane_parts) for plane_parts in gmsh_curves]
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        #for e in zip(box_surfaces, ov):
            #print("parent " + str(e[0]) + " -> child " + str(e[1]))
        #print("ov = ",len(ov),ov)
        #print("ovv = ",len(ovv),ovv)
        
        #print("Box Surfaces = ", len(box_surfaces), box_surfaces)
        
        # Identify all the dim3 volumes that were touched by the fragment
        fraged_vols = filter_tuples_by_first_entry(ovv, 3)
        
        # Filter out the volumes that did not exist before the fragment (only new vols)
        new_vols = unique_points(fraged_vols,current_vols)
        
        #repeat the current stratigraphy name that we are cutting
        vol_namer = [vol_name] * len(new_vols)
        
        #print("new vols: ", new_vols, "vol_namer: ", vol_namer)
        # New elements inherit their parent names
        tracker.update_entries(box_surfaces+all_surf, ov)
        # New volumes are given the new stratigraphy name
        tracker.add_entry(new_vols, vol_namer)   


    """_____________________________  FAULT FRAGMENTING LOOP __________________________________"""
    # ORDER IS LESS OF AN ISSUE FOR THIS AS WE PROPAGATE VOLUME NAMES BY STRATIGRAPHY, 
    # FOR CLEAN ORDERING, CONVENTION SHOULD BE FRAGMENT WEST TO EAST

    for surface, name in zip(all_faults, fault_surface_names):
        """Adding surfaces to OpenCascade"""

        points = surface.points
        faces = surface.faces.reshape((-1, 4))[:, 1:]
        #points.round(2), faces 

        gmsh_points = []
        #points = [(0,0,.5), (0,1,.5), (1,1,.5), (1,0,.5)]
        for pt in points:
            gmsh_points.append(gmsh.model.occ.add_point(pt[0], pt[1], pt[2]))
        #print("Points ",gmsh_points)

        gmsh_curves = []
        for face in faces:
                lines = []
                for j in range(len(face)):
                    p1 = gmsh_points[face[j]]
                    p2 = gmsh_points[face[(j + 1) % len(face)]]
                    lines.append(gmsh.model.occ.add_line(p1, p2))
                    #print(f"line {lines} {j}")
                loop = gmsh.model.occ.add_curve_loop(lines)
                #print("loop ", loop)
                surface_tag = gmsh.model.occ.add_plane_surface([loop])
                #print("adding surface tag : ", surface_tag)
                gmsh_curves.append(surface_tag)
                gmsh.model.occ.synchronize()
        #print("Curves :", gmsh_curves)
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        namer = [name] * len(gmsh_curves)
        #print(namer)
        tracker.add_entry([(2, plane_parts) for plane_parts in gmsh_curves], namer)

        
    
        #all_2dim = gmsh.model.occ.get_entities(dim=2) #box creation order (ZY- , ZY+ , ZX- , ZX+ , XY- , XY+) or (W, E, S, N, Dn, Up)
        #print("Box 2 dim tags tuples: ", all_2dim)

        box_3dim = gmsh.model.occ.get_entities(dim=3) #original box
        

        #box_2dim = gmsh.model.occ.get_entities(dim=2)
        #boxtags = [tags[1] for tags in box_2dim]

        #surf_dict = "surf"

        #print("Box tags without dimension: ",boxtags)

        # return all dim2 surface elements from the current model
        all_surf = gmsh.model.occ.get_entities(dim=2)
    
        #print("gmsh _curves ",gmsh_curves)
        
        #combine all the elements in the volume, we need them all so that the fragment operation reports all the parent child relationships correctly without gaps
        box_surfaces = all_surf+box_3dim
        if debug:
            print(f"Cutting surface, {name} - All box surfaces/Vols: ", box_surfaces, "box_3dim: ", all_surf)
        
        ovv, ov = gmsh.model.occ.fragment(box_surfaces, all_surf, removeTool=True, removeObject=True) #[(2, plane_parts) for plane_parts in gmsh_curves]
        #gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        #for e in zip(box_surfaces, ov):
            #print("parent " + str(e[0]) + " -> child " + str(e[1]))
        #print("ov = ",len(ov),ov)
        #print("ovv = ",len(ovv),ovv)
        
        #print("Box Surfaces = ", len(box_surfaces), box_surfaces)
        
        # Identify all the dim3 volumes that were touched by the fragment
        #original_vols = filter_tuples_by_first_entry(ov, 3)
        #fraged_vols = filter_tuples_by_first_entry(ovv, 3)
        #for e in zip(fraged_vols, original_vols):
        #    print("Volume parent " + str(e[0]) + " -> child " + str(e[1]))
        # Filter out the volumes that did not exist before the fragment (only new vols)
        #new_vols = msh.unique_points(fraged_vols, current_vols)
        
        #repeat the current stratigraphy name that we are cutting
        #vol_namer = [vol_name] * len(new_vols)
        
        #print("new vols: ", new_vols, "vol_namer: ", vol_namer)
        # New elements inherit their parent names
        comb_surf = box_surfaces+all_surf
        for e in zip(comb_surf, ov):
            if debug:
                print("parent " + str(e[0]) + " -> child " + str(e[1]))
        tracker.update_entries(comb_surf[::-1], ov[::-1])
        # New volumes are given the new stratigraphy name
        #tracker.add_entry(original_vols, fraged_vols)   



    counts = tracker.get_name_counts(by_dimension=True)
    #counts[2]

    tags_3dim = gmsh.model.occ.get_entities(dim=3)
    tags_2dim = gmsh.model.occ.get_entities(dim=2)
    #fragtags_dim2 = msh.filter_tuples_by_first_entry(ovv, 2)


    #print('Dim3 Tags: ', tags_3dim)
    dim3_names_list = []
    for tag in tags_3dim:
        #print('current tag 3', tag)
        gmsh.model.add_physical_group(3, [tag[1]], tag=tag[1])
        gmsh.model.set_physical_name(3, tag[1], f"{tracker.lookup_name(tag)}_{tag[1]:04d}")
        dim3_names_list.append(f"{tracker.lookup_name(tag)}_{tag[1]:04d}")
        counts[3][tracker.lookup_name(tag)] = counts[3][tracker.lookup_name(tag)]-1 #volumes physical group name is numbered by the remaining volumes that name, should count back to zero
    #print("Dim 2 frag tags: ",tags_2dim)
    dim2_names_list = []
    for tag in tags_2dim:
        #print('current tag 2', tag)
        gmsh.model.add_physical_group(2, [tag[1]], tag=tag[1])
        #gmsh.model.set_physical_name(2, tag[1], f"{tracker.lookup_name(tag)}_{counts[2][tracker.lookup_name(tag)]+10000}") depreciated
        gmsh.model.set_physical_name(2, tag[1], f"{tracker.lookup_name(tag)}_{counts[2][tracker.lookup_name(tag)]:04d}")
        dim2_names_list.append(f"{tracker.lookup_name(tag)}_{counts[2][tracker.lookup_name(tag)]:04d}")
        counts[2][tracker.lookup_name(tag)] = counts[2][tracker.lookup_name(tag)]-1 #surfaces physical group name is numbered by the remaining surfaces with that name, should count back to zero

    dim2_sorted_surfaces = sort_by_prefix(all_dim2_prefix, dim2_names_list)
    dim3_sorted_surfaces = sort_by_prefix(volume_names, dim3_names_list)
    #print(counts)
    gmsh.model.occ.removeAllDuplicates()  ##### TESTING - Added for PLC error #### 
    gmsh.model.occ.synchronize()
    pnt_entities = gmsh.model.occ.get_entities(dim=0)
    gmsh.model.mesh.setSize(pnt_entities, fieldsize)
    #gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
    #gmsh.option.setNumber("Mesh.LcIntegrationPrecision", 1.e-4)
    #gmsh.option.setNumber("Mesh.ToleranceInitialDelaunay", 1e-4)
    #gmsh.option.setNumber("Geometry.Tolerance", 1e-4)
    #gmsh.option.setNumber("Geometry.ToleranceBoolean", 1e-4)
    #gmsh.option.setNumber("Geometry.OCCSewTolerance", 1e-3)
    gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
    #gmsh.option.setNumber("Geometry.OCCSewTolerance", 1e-8)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
    #gmsh.option.setNumber("Geometry.OCCFixDegenerated", 5)
    #gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 5)
    #gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 5)
    #gmsh.option.setNumber("Geometry.OCCSewFaces", 5)
    #gmsh.option.setNumber("Geometry.AutoCoherence", 5) 
    #gmsh.option.setNumber("Mesh.MeshSizeMin", .3)
    if mesh:
        gmsh.model.mesh.generate(Dim)
    gmsh.write(output_file)
    gmsh.finalize()

    return dim2_sorted_surfaces, dim3_sorted_surfaces


def geo_mesh_builder(all_horizons=[], volume_names=[], vol_surface_names=[], all_faults=[], 
                             fault_surface_names=[], model_name="divided_volume", output_file='./mesher.msh', 
                             bounding_box=[0,1,0,1,0,1], fieldsize=500, rotation=0.0, rot_center='origin', Dim=3, debug=True, mesh=False, track_names=True):
    
    """
    Finite Element Method (FEM) mesh constructor from "watertight" polydata surfaces.

    Parameters:
    all_horizons [list of obj]: pyvista Polydata surfaces for dividing the mesh, from bottom to top of the mesh
    volume_names [list of str]: names of volumes, must be in corresponding order to all_horizon parameter with a leading name that indicates the bottom volume e.g 'basement'
    vol_surface_names [list of str.]: names of horizon surfaces, in all_surfaces

    all_faults [list of obj]: pyvista Polydata surfaces for dividing the mesh, these surfaces will not result in volume renaming
    fault_surface_names [list of str.]: names of fault surfaces, in all_faults

    model_name [str]: gmesh mesh name 
    output_file [str]: filepath for saving out the completed mesh

    bounding_box [list]: extents of box volume  [ xmin, xmax, ymin, ymax, zmin, zmax ]

    fieldsize [int]: Mesh fieldsize parameter

    rotation [float]: Degrees of rotation (CW from North or y-axis) for output boundary

    rot_center [tuple]: point, as (X,Y,Z), to rotate the output geometry. (keyword cases: "origin" = (0,0,0) , "centroid" = centroid of box geometry)

    Dim [int]: 1D, 2D, 3D mesh output parameter

    debug [bool]: supress debug output

    mesh [bool]: generate mesh before writing file   

    Outputs: dim2 and dim3 element list of lists of the surfaces sorted by their name prefix

    """ 

    def fragment_volume(surface, name, vol_name, debug=debug, do_not_rename_divided_volumes=False):
        
            """
            GMSH Fragmenting routine,
            """
            
        
            current_vols = gmsh.model.occ.get_entities(dim=3)

            """Adding surfaces to OpenCascade"""

            points = surface.points
            faces = surface.faces.reshape((-1, 4))[:, 1:]
            theta = math.radians(rotation) 
            max_tag = gmsh.model.occ.getMaxTag(2)
            
            gmsh.model.occ.setMaxTag(2, max_tag+100)
            gmsh_curves = list()
            gmsh_points = list()
            for pt in points:
                if abs(rotation) > 1e-12:
                    incremental_tags = gmsh.model.occ.get_entities(dim=0)
                    #print("incremental tags ", incremental_tags)
                pointTag = gmsh.model.occ.add_point(pt[0], pt[1], pt[2])
                #print("Point Tag ", pointTag)
                if abs(rotation) > 1e-12:
                    gmsh.model.occ.rotate(
                        [(0, pointTag)],     # 3 = volume dimension
                        cx, cy, cz,      # center of rotation
                        0, 0, 1,         # rotation axis (Z)
                        theta
                        )
                    current_tags = gmsh.model.occ.get_entities(dim=0)
                    new_tag = unique_points(current_tags, incremental_tags)
                    if debug:
                        print("current_tags", current_tags, "new tag ", new_tag[0][1])
                    gmsh_points.append(new_tag[0][1])
                else:
                    gmsh_points.append(pointTag)

            gmsh_curves = []
            for face in faces:
                    lines = []
                    for j in range(len(face)):
                        p1 = gmsh_points[face[j]]
                        p2 = gmsh_points[face[(j + 1) % len(face)]]
                        lines.append(gmsh.model.occ.add_line(p1, p2))
                
                    loop = gmsh.model.occ.add_curve_loop(lines)

                    surface_tag = gmsh.model.occ.add_plane_surface([loop])
            
                    gmsh_curves.append(surface_tag)
                    gmsh.model.occ.synchronize()
        
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            
            namer = [name] * len(gmsh_curves)
            tracker.add_entry([(2, plane_parts) for plane_parts in gmsh_curves], namer)

            # return all dim2 surface elements from the current model
            box_3dim = gmsh.model.occ.get_entities(dim=3) 
            all_surf = gmsh.model.occ.get_entities(dim=2)
        
            #combine all the elements in the volume, we need them all so that the fragment operation reports all the parent child relationships correctly without gaps
            box_surfaces = all_surf+box_3dim
            if debug:
                print(f"Cutting surface, {name} - All box surfaces/Vols: ", box_surfaces, "box_3dim: ", box_3dim)
            
            

            ovv, ov = gmsh.model.occ.fragment(box_surfaces, all_surf, removeTool=True, removeObject=True) #[(2, plane_parts) for plane_parts in gmsh_curves]
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            
            if debug:
                for e in zip(box_surfaces, ov):
                    print("parent " + str(e[0]) + " -> child " + str(e[1]))
                print("ov = ",len(ov),ov)
                #print("ovv = ",len(ovv),ovv)
            comb_surf = box_surfaces+all_surf
            
            print("Total Renames OV ", len(ov))
            for i in zip(comb_surf, ov):
                print("Parent name --> ", i[0], tracker.lookup_name(i[0]), " Childname --> ", i[1])
            print("--------------------------")
            print("Total Renames OVV ", len(ovv))
            for i in zip(box_surfaces, ovv):
                print("Parent name --> ", i[0], tracker.lookup_name(i[0]), " Childname --> ", i[1])
            print("--------------------------")
            #for e in zip(box_surfaces, ov):
            #print("parent " + str(e[0]) + " -> child " + str(e[1]))
            
            clip = comb_surf[0]
            i = 1
            while comb_surf[i] != clip:
                i = i+1


            
            print("RENAMING ",i," fragments")
            for j in zip(comb_surf[:i], ov[:i]):
                print("Parent name --> ", j[0], tracker.lookup_name(j[0]), " Childname --> ", j[1])
            print("--------------------------")
            
            if do_not_rename_divided_volumes:
                print("ONLY RENAMING SURFACES")
                
                #for e in zip(comb_surf, ov):
                    #if debug:
                    #    print("parent " + str(e[0]) + " -> child " + str(e[1]))
                        #print(f"Combined Surface: ", "ov ", ov)
                #tracker.update_entries(comb_surf[::-1], ov[::-1])
                tracker.update_entries(comb_surf[:i][::-1], ov[:i][::-1])


            elif do_not_rename_divided_volumes == False:
                print("RENAMING VOLUMES AND SURFACES")
                # Identify all the dim3 volumes that were touched by the fragment
                fraged_vols = filter_tuples_by_first_entry(ovv, 3)
                
                # Filter out the volumes that did not exist before the fragment (only new vols)
                new_vols = unique_points(fraged_vols,current_vols)
                if debug:
                    print("fraged vols / current vols: ", fraged_vols, current_vols)
                #repeat the current stratigraphy name that we are cutting
                vol_namer = [vol_name] * len(new_vols)
                if debug:
                    print("vol namer", vol_namer, "new vols ", new_vols)
                comb_surf = box_surfaces+all_surf
                
                #print("new vols: ", new_vols, "vol_namer: ", vol_namer)
                # New elements inherit their parent names
                tracker.update_entries(comb_surf[:i][::-1], ov[:i][::-1])
                # New volumes are given the new stratigraphy name
                tracker.add_entry(new_vols, vol_namer)
                


                # Identify all the dim3 volumes that were touched by the fragment
        # Identify all the dim3 volumes that were touched by the fragment
        #fraged_vols = filter_tuples_by_first_entry(ovv, 3)
        
        # Filter out the volumes that did not exist before the fragment (only new vols)
        #new_vols = unique_points(fraged_vols,current_vols)
        
        #repeat the current stratigraphy name that we are cutting
        #vol_namer = [vol_name] * len(new_vols)
        
        #print("new vols: ", new_vols, "vol_namer: ", vol_namer)
        # New elements inherit their parent names
        #tracker.update_entries(box_surfaces+all_surf, ov)
        # New volumes are given the new stratigraphy name
        #tracker.add_entry(new_vols, vol_namer)  
            """

            fraged_vols = filter_tuples_by_first_entry(ovv, 3)
            
            # Filter out the volumes that did not exist before the fragment (only new vols)
            new_vols = unique_points(fraged_vols,current_vols)
            
            #repeat the current stratigraphy name that we are cutting
            vol_namer = [vol_name] * len(new_vols)
            
            #print("new vols: ", new_vols, "vol_namer: ", vol_namer)
            # New elements inherit their parent names
            tracker.update_entries(box_surfaces+all_surf, ov)
            # New volumes are given the new stratigraphy name
            tracker.add_entry(new_vols, vol_namer) 
            """

    if rot_center=='centroid':
        # Compute box center
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        cz = 0.5 * (zmin + zmax)

    if rot_center=='origin':
        cx=0.
        cy=0.
        cz=0.
    
    if type(rot_center) == tuple:
        cx=rot_center[0]
        cy=rot_center[1]
        cz=rot_center[2]

    xmin, xmax, ymin, ymax, zmin, zmax = bounding_box
    extents = (xmin, (xmax-xmin), ymin, (ymax-ymin), zmin, (zmax-zmin))
    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.model.add(model_name)

        # Define a rectangular box in Gmsh using OCC kernel
        #length, width, height = box_dimensions
        #box = gmsh.model.occ.add_box(0, 0, 0, length, width, height)

    #box_tags =gmsh.model.occ.add_box(extents[0], extents[2], extents[4], extents[1], extents[3], extents[5])

    box_tags, center = create_rotated_box(xmin, xmax, ymin, ymax, zmin, zmax, rot_center=rot_center, azimuth_deg=rotation, tag=1)

    if debug:
        print("box tags ",box_tags, " center: ",rot_center, "Rotation Azimuth Deg: ", rotation)
    #box = gmsh.model.occ.get_entities(dim=3)
        # Synchronize after adding the box
       
    #face_names =  ['West', 'East', 'South', 'North', 'Base', 'Top'] #box surfaces creation order (ZY- , ZY+ , ZX- , ZX+ , XY- , XY+) or (W, E, S, N, Dn, Up)
    face_names = rotated_box_face_directions(rotation)
    #collect all the surfaces naming components to sort dim2 element names after fragmenting
    all_dim2_prefix = vol_surface_names+fault_surface_names+face_names
    
    
    tracker = DimensionIDTracker()
    
    gmsh.model.occ.synchronize()
    ### Retrive 2D/3D entities 
    box = gmsh.model.occ.get_entities(dim=3)
    box_2dim = gmsh.model.occ.get_entities(dim=2)

    ### BOUNDARY SURFACE NAMING
    if debug:
        print("Box Tag ", box, [f"box_2_dim tags: {tags}, " for tags in zip(box_2dim, face_names)])
    tracker.add_entry(box_2dim, face_names)
    
    ### INITAL VOLUME NAMING
    if debug:
        print("Tracker: Volume BOX", box[0], "Named: ", volume_names[0])
    tracker.add_entry([box[0]], [volume_names[0]]) 

 

    """_____________________________  STRATIGRAPHY FRAGMENTING LOOP __________________________________"""
    # FIRST WE CUT THE BLOCK FROM THE BOTTOM UP WITH STRATIGRAPHY (FAULTS ARE HANDLED IN THE NEXT LOOP)
    for surface, name, vol_name in zip(all_horizons, vol_surface_names, volume_names[1:]):
        print("###### CUTTING STRAT SURFACE ", surface,"named", name, "with volume name ", vol_name)
        #volume = gmsh.model.occ.get_entities(dim=3)
        new_vols = fragment_volume(surface, name, vol_name, debug=debug, do_not_rename_divided_volumes=False)
        if debug:
            print(new_vols)
    """_____________________________  FAULT FRAGMENTING LOOP __________________________________"""
    # ORDER IS LESS OF AN ISSUE FOR THIS AS WE PROPAGATE VOLUME NAMES BY STRATIGRAPHY, 
    # FOR CLEAN ORDERING, CONVENTION SHOULD BE FRAGMENT WEST TO EAST
    """
    for surface, name in zip(all_faults, fault_surface_names):
        #Adding surfaces to OpenCascade

        points = surface.points
        faces = surface.faces.reshape((-1, 4))[:, 1:]
        #points.round(2), faces 

        gmsh_points = []
        #points = [(0,0,.5), (0,1,.5), (1,1,.5), (1,0,.5)]
        for pt in points:
            gmsh_points.append(gmsh.model.occ.add_point(pt[0], pt[1], pt[2]))
        #print("Points ",gmsh_points)

        gmsh_curves = []
        for face in faces:
                lines = []
                for j in range(len(face)):
                    p1 = gmsh_points[face[j]]
                    p2 = gmsh_points[face[(j + 1) % len(face)]]
                    lines.append(gmsh.model.occ.add_line(p1, p2))
                    #print(f"line {lines} {j}")
                loop = gmsh.model.occ.add_curve_loop(lines)
                #print("loop ", loop)
                surface_tag = gmsh.model.occ.add_plane_surface([loop])
                #print("adding surface tag : ", surface_tag)
                gmsh_curves.append(surface_tag)
                gmsh.model.occ.synchronize()
        #print("Curves :", gmsh_curves)
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        namer = [name] * len(gmsh_curves)
        #print(namer)
        tracker.add_entry([(2, plane_parts) for plane_parts in gmsh_curves], namer)

        
    
        #all_2dim = gmsh.model.occ.get_entities(dim=2) #box creation order (ZY- , ZY+ , ZX- , ZX+ , XY- , XY+) or (W, E, S, N, Dn, Up)
        #print("Box 2 dim tags tuples: ", all_2dim)
        # return all dim2 surface elements from the current model
        box_3dim = gmsh.model.occ.get_entities(dim=3) #original box
        all_surf = gmsh.model.occ.get_entities(dim=2)
    
        #print("gmsh _curves ",gmsh_curves)
        
        #combine all the elements in the volume, we need them all so that the fragment operation reports all the parent child relationships correctly without gaps
        box_surfaces = all_surf+box_3dim
        if debug:
            print(f"Cutting surface, {name} - All box surfaces/Vols: ", box_surfaces, "box_3dim: ", all_surf)
        
        ovv, ov = gmsh.model.occ.fragment(box_surfaces, all_surf, removeTool=True, removeObject=True) #[(2, plane_parts) for plane_parts in gmsh_curves]
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        gmsh.model.occ.removeAllDuplicates()
        gmsh.model.occ.synchronize()
        #for e in zip(box_surfaces, ov):
            #print("parent " + str(e[0]) + " -> child " + str(e[1]))

        comb_surf = box_surfaces+all_surf
        for e in zip(comb_surf, ov):
            if debug:
                print("parent " + str(e[0]) + " -> child " + str(e[1]))
        tracker.update_entries(comb_surf[::-1], ov[::-1])
        # New volumes are given the new stratigraphy name
        #tracker.add_entry(original_vols, fraged_vols)   
    """
    for surface, name, vol_name in zip(all_faults, fault_surface_names, volume_names[1:]):
        print("CUTTING FAULT SURFACE ", surface,"named", name, "with volume name ", vol_name)
        #volume = gmsh.model.occ.get_entities(dim=3)
        new_vols = fragment_volume(surface, name, vol_name, debug=debug, do_not_rename_divided_volumes=True)
        if debug:
            print(new_vols)


    
    

    """ WRITING NAMES STORED IN TRACKER TO THE OCC ENTITIES  """

    """
    if abs(rotation) > 1e-12:
        vol_tags = gmsh.model.occ.get_entities(dim=3)
        theta = math.radians(rotation) 
        gmsh.model.occ.rotate(
                    [(2,9)],         # 3 = volume dimension
                    cx, cy, cz,      # center of rotation
                    0, 0, 1,         # rotation axis (Z)
                    theta
                )
        gmsh.model.occ.synchronize()
    """
    #if abs(rotation) > 1e-12:
    #    pre_rot_pnt_tags = gmsh.model.occ.get_entities(dim=0)
    #    pre_rot_line_tags = gmsh.model.occ.get_entities(dim=1)
    #    pre_rot_surf_tags = gmsh.model.occ.get_entities(dim=2)
    #    theta = math.radians(rotation) 
        # rotation around Z-axis
        #print(f"Dim 3 tags {vol_tags}")
    #    for i in (2,3):
        
    #        print("Dim " ,i)
    #        vol_tags = gmsh.model.occ.get_entities(dim=i)
    #        print(f"Dim {i} tags {vol_tags}")
    #        tags = filter_tuples_by_first_entry(vol_tags, i)
    #        print(f"Tags {tags}")
    #        for j in tags:
    #            incremental_tags = gmsh.model.occ.get_entities(dim=i)
    #            print("incremental tags ", incremental_tags)
    #            rot_out = gmsh.model.occ.rotate(
    #                [(i, j[1])],     # 3 = volume dimension
    #                cx, cy, cz,      # center of rotation
    #                0, 0, 1,         # rotation axis (Z)
    #                theta
    #                )
                    #gmsh.model.occ.removeAllDuplicates()
                #gmsh.model.occ.synchronize()
    #            print(rot_out)
    #            current_tags = gmsh.model.occ.get_entities(dim=i)
    #            print("current_tags", current_tags)
    #            new_tag = unique_points(current_tags, incremental_tags)
    #            print(f"Old tag: {[(i,j[1])]}", "new_tag:", new_tag)
    #            if new_tag != []:
    #                tracker.update_entries([(i,j[1])], [new_tag])
    #    gmsh.model.occ.remove(pre_rot_surf_tags, recursive=True)
    #    gmsh.model.occ.remove(pre_rot_line_tags, recursive=True)
    #    gmsh.model.occ.remove(pre_rot_pnt_tags, recursive=True)
        #gmsh.model.occ.synchronize()
        
    #    gmsh.model.occ.synchronize()
        #gmsh.model.occ.removeAllDuplicates()
    #    print("dim0 ", gmsh.model.occ.get_entities(dim=0))
    #    print("dim1 ", gmsh.model.occ.get_entities(dim=1))
    #    print("dim2 ", gmsh.model.occ.get_entities(dim=2))

    if track_names:
        counts = tracker.get_name_counts(by_dimension=True)
        if debug:
            print(counts)

        tags_3dim = gmsh.model.occ.get_entities(dim=3)
        tags_2dim = gmsh.model.occ.get_entities(dim=2)
        #fragtags_dim2 = msh.filter_tuples_by_first_entry(ovv, 2)


        #print('Dim3 Tags: ', tags_3dim)
        dim3_names_list = []
        for tag in tags_3dim:
            #print('current tag 3', tag)
            gmsh.model.add_physical_group(3, [tag[1]], tag=tag[1])
            gmsh.model.set_physical_name(3, tag[1], f"{tracker.lookup_name(tag)}_{tag[1]:04d}")
            if debug:
                print(f"{tracker.lookup_name(tag)}_{tag[1]:04d}", "->", tag[1])
            dim3_names_list.append(f"{tracker.lookup_name(tag)}_{tag[1]:04d}")
            counts[3][tracker.lookup_name(tag)] = counts[3][tracker.lookup_name(tag)]-1 #volumes physical group name is numbered by the remaining volumes that name, should count back to zero
        #print("Dim 2 frag tags: ",tags_2dim)
        dim2_names_list = []
        for tag in tags_2dim:
            #print('current tag 2', tag)
            gmsh.model.add_physical_group(2, [tag[1]], tag=tag[1])
            #gmsh.model.set_physical_name(2, tag[1], f"{tracker.lookup_name(tag)}_{counts[2][tracker.lookup_name(tag)]+10000}") depreciated
            if debug:
                print("naming tag ", tag)
            gmsh.model.set_physical_name(2, tag[1], f"{tracker.lookup_name(tag)}_{counts[2][tracker.lookup_name(tag)]:04d}")
            dim2_names_list.append(f"{tracker.lookup_name(tag)}_{counts[2][tracker.lookup_name(tag)]:04d}")
            counts[2][tracker.lookup_name(tag)] = counts[2][tracker.lookup_name(tag)]-1 #surfaces physical group name is numbered by the remaining surfaces with that name, should count back to zero


        # Returns surfaces/Volumes for addressing FEM input files
        dim2_sorted_surfaces = sort_by_prefix(all_dim2_prefix, dim2_names_list)
        dim3_sorted_surfaces = sort_by_prefix(volume_names, dim3_names_list)
    
    

    # Clear duplicate points (if they exist)
    gmsh.model.occ.removeAllDuplicates()  ##### TESTING - Added for PLC error #### 
    gmsh.model.occ.synchronize()

    # Set Field Size parameter for the mesh resolution
    pnt_entities = gmsh.model.occ.get_entities(dim=0)
    gmsh.model.mesh.setSize(pnt_entities, fieldsize)

    # Set Geometry tolerances
    gmsh.option.setNumber("Geometry.Tolerance", 1e-8)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)
    gmsh.option.setNumber("Geometry.OCCSewFaces", 1)


    # Mesh Generation with dimension var
    if mesh:
        gmsh.model.mesh.generate(Dim)
    gmsh.write(output_file)
    gmsh.finalize()
    hierarchy = tracker.get_hierarchy()
    if track_names:
        return dim2_sorted_surfaces, dim3_sorted_surfaces, hierarchy