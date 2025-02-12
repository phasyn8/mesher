###### AN AUTOMATED MESHING TOOL FOR FEM SIMULATIONS ######
# 
# ACCEPTS VTK surfaces and parses them with OpenCascadeCAD/gmsh to create completed 3D meshes
# for use with MOOSE or other FEM frameworks
import pyvista as pv
import gmsh
import numpy as np
import pandas as pd

""" Worker functions NEED TO IMPLEMENT INTO A WOKRING CLASS """

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
    unique_points_list = list(map(list, unique_points_set))
    
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