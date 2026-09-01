using Mantis

ref_geo_h2(g) = Geometry.refinement_uniform(g, 2)
rel_1 = Hierarchical.Relations(i -> ((i - 1) * 2 + 1, i * 2), i -> (div(i + 1, 2),))
geo_l1 = Geometry.create_cartesian_box((0.0,), (1.0,), (2,))
geo_l2 = Hierarchical.Refinement(geo_l1, ref_geo_h2)()
geo_l3 = Hierarchical.Refinement(geo_l2, ref_geo_h2)()

scal_1 = Hierarchical.Scaling(geo_l1, geo_l2, rel_1)
scal_2 = Hierarchical.Scaling(geo_l2, geo_l3, rel_1)

marked_elements_per_level = [[1], [3]]
active_info = Hierarchical.ActiveInfo(marked_elements_per_level)

hgeo = Geometry.HierarchicalGeometry(Hierarchical.NestedHierarchy(active_info, scal_1))
