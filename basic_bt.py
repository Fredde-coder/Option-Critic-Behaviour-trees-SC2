import py_trees
from BT_behaviours.behaviours import *

root = py_trees.composites.Selector(name="Tree")
b_barracks = py_trees.composites.Selector(name="Build barracks")
q_marine = py_trees.composites.Sequence(name="Queue marine")
b_supply_depot = py_trees.composites.Sequence(name="Offset supply demand")
b_scv = py_trees.composites.Sequence(name='\build_scvs')

Multiple_barracks = py_trees.composites.Sequence(name="Multiple_barracks")
Multiple_barracks.add_children([Check_barracks(name="barracks exists"),Build_barrack()])
first_barrack = py_trees.composites.Sequence(name="first barrack")
first_barrack.add_children([Check_barracks(name="barracks don't exist"),Build_barrack()])
b_barracks.add_children([Multiple_barracks,first_barrack])

q_marine.add_children([Check_marines(name="all barrack queues less than three"), Build_marine()])

b_supply_depot.add_children([Check_supply(name="supply-used<Barracks"),Build_supply_depot()])

b_scv.add_children([Check_barracks(name="barracks exists"),Build_scv()])
root.add_children([b_barracks, q_marine, b_supply_depot,b_scv])

print(py_trees.display.ascii_tree(root))
print(py_trees.display.dot_tree(root))





py_trees.display.render_dot_tree(root,target_directory="Bt_visualized")
