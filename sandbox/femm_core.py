### packages - 
# pip install pyfemm
# also install femm software
# it must already be installed in bintu


import matplotlib.pyplot as plt
import matplotlib.tri as tri
from pathlib import Path
import femm

current_dir = Path('.')
femm.openfemm()

model_filename = 'c_core.fem'
model_filepath = current_dir / model_filename
print(model_filepath.absolute())

# femm.opendocument(model_filepath.absolute());
femm.newdocument(0);

#(-18,30)                  (0,30)            (6,30)
#|-----------------------------      ----------
#|                            |     |         |
#|                            |     |         |
#|                            |     |         |
#|       |--------------------|     |         |
#|       |(-12,24)          (0,24)  |         |                     
#|       |                          |         |                     
#|       |                          |         |                     
#|       |                  (0,6)   |         |
#|       |--------------------|     |         |
#|                            |     |         |
#|                            |     |         |
#|                            |     |         |
#-----------------------------|     |---------|
#(-18,0)                   (0,0)   (3,0)     (6,0)



femm.mi_getmaterial('Air')
femm.mi_addmaterial('LinearIron', 2100, 2100, 0, 0, 0, 0, 0, 1, 0, 0, 0)
femm.mi_getmaterial('Cold rolled low carbon strip steel')            
femm.mi_getmaterial('Copper')
femm.mi_addcircprop('icoil', 1, 1);

femm.mi_clearselected();

femm.mi_drawline(0, 0, -18, 0)
femm.mi_drawline(-18, 0, -18, 30)
femm.mi_drawline(-18, 30, 0, 30)
femm.mi_drawline(0, 30, 0, 24)
femm.mi_drawline(0, 24, -12, 24)
femm.mi_drawline(-12, 24, -12, 6)
femm.mi_drawline(-12, 6, 0, 6)
femm.mi_drawline(0, 6, 0, 0)

femm.mi_addblocklabel(-15, 15);

# Apply the materials to the appropriate block labels
femm.mi_selectlabel(-15, 15);
femm.mi_setblockprop('LinearIron', 0, 0.5, '<None>', 0, 5, 0);
femm.mi_clearselected()

## Coil
coil_start_x = -12
coil_start_y = 6
coil_width = 3
coil_length = 18

femm.mi_drawrectangle(coil_start_x, coil_start_y, coil_start_x + coil_width, coil_start_y + coil_length)
femm.mi_addblocklabel(coil_start_x + coil_width/2, coil_start_y + coil_length/2)
femm.mi_clearselected();
femm.mi_selectlabel(coil_start_x + coil_width/2, coil_start_y + coil_length/2)
femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, 500); 
femm.mi_clearselected();


coil2_start_x = coil_start_x - 3*coil_width

femm.mi_drawrectangle(coil2_start_x, coil_start_y, coil2_start_x + coil_width, coil_start_y + coil_length)
femm.mi_addblocklabel(coil2_start_x + coil_width/2, coil_start_y + coil_length/2)
femm.mi_clearselected();

femm.mi_selectlabel(coil2_start_x + coil_width/2, coil_start_y + coil_length/2)
femm.mi_setblockprop('Copper', 0, 0, 'icoil', 0, 1, -500); 
femm.mi_clearselected();

### Armature
arm_start_x = 3
arm_start_y = 0
arm_width = 6
arm_length = 30
femm.mi_drawrectangle(arm_start_x, arm_start_y, arm_start_x + arm_width, arm_start_y + arm_length)

femm.mi_addblocklabel(arm_start_x + arm_width/2, arm_start_y + arm_length/2)
femm.mi_selectlabel(arm_start_x + arm_width/2, arm_start_y + arm_length/2)
femm.mi_setblockprop('LinearIron', 0, 0.5, '<None>', 0, 5, 0);
femm.mi_clearselected();

### Air box

femm.mi_drawrectangle(-100, 100, 100, -100)
femm.mi_addblocklabel(-65, 75);
femm.mi_selectlabel(-45, 95);
femm.mi_setblockprop('Air', 50, 0, '<None>', 0, 0, 0);
femm.mi_clearselected();

### Dirichlet BC

femm.mi_selectsegment(-95, 0)
femm.mi_selectsegment(0, 95)
femm.mi_selectsegment(95, 0)
femm.mi_selectsegment(0, -95)
femm.mi_addboundprop('dirichlet', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
femm.mi_setsegmentprop('dirichlet', 0, 0, 0, 0)
femm.mi_clearselected();  


femm.mi_zoomnatural();
femm.mi_saveas(str(model_filepath.absolute()));
# femm.opendocument(model_filepath.absolute());

femm.mi_clearselected();
femm.mi_zoomnatural()
femm.mi_analyse(0)
femm.mi_loadsolution()


## potential at all nodes
potential_nodes, x_nodes, y_nodes = [], [], []
num_nodes = femm.mo_numnodes()
for i in range(1, num_nodes):
    [x, y] = femm.mo_getnode(i)
    x_nodes.append(x)
    y_nodes.append(y)
    potential_nodes.append(femm.mo_geta(x,y))


fig, ax1 = plt.subplots(1,1,figsize=(30,10))

## True Exact
# ax2.tricontour(x, y, z, levels=20, linewidths=0.5, colors='k')
cntr1 = ax1.tricontourf(x_nodes, y_nodes, potential_nodes, levels=20, cmap="jet")
fig.colorbar(cntr1, ax=ax1)

plt.show()


## getting magnetic potential on armature interface
x = 3
arm_air_i = []
for y in range(0,31):
    arm_air_i.append(femm.mo_geta(x,y))


