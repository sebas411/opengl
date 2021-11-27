# Software renderer using opengl

## Installation
* Clone repository
* Open a terminal and install the needed modules for python using the following command:
```pip install -r requirements.txt```

## Usage
This renderer contains 3 different shaders. The first one is the equivalent to the Gouraud shader, the second one uses the vertex normals as colors and the third twist the object as time passes.  
You can change the current shader in line 151 of the code in o.py choosing between the variables for shader1, shader2 and shader3.  

### Controls:
```
w,a,s,d: move around
spacebar, shift: go up or down
arrow keys: change orientation of the camera
```
