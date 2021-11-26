import pygame
import numpy
from obj import *
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glm
from math import cos, pi, sin

pygame.init()
screen = pygame.display.set_mode((1200, 720), pygame.OPENGL | pygame.DOUBLEBUF)
glClearColor(0, 0, 0, 1.0)
glEnable(GL_DEPTH_TEST)
clock = pygame.time.Clock()
pygame.key.set_repeat(1)

vertex_shader = """
#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 texcoords;

uniform mat4 theMatrix;
uniform mat4 rotMatrix;
uniform vec3 light;

out vec2 vertexTexcoords;
out float intensity;

vec3 transnormal = normalize(rotMatrix * vec4(normal.x, normal.y, normal.z, 1)).xyz;
void main() 
{
  intensity = dot(transnormal, normalize(light));
  gl_Position = theMatrix * vec4(position.x, position.y, position.z, 1);
  vertexTexcoords = texcoords.xy;
}
"""

vertex_shader2 = """
#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 texcoords;

uniform mat4 theMatrix;
uniform mat4 rotMatrix;
uniform vec3 light;
uniform int clock;

out vec3 color;
out float intensity;

vec3 transnormal = normalize(rotMatrix * vec4(normal.x, normal.y, normal.z, 1)).xyz;
void main() 
{
  intensity = dot(normal, normalize(light));
  gl_Position = theMatrix * vec4(position.x, position.y, position.z, 1);
  color = transnormal;
}
"""

vertex_shader3 = """
#version 460

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec3 texcoords;

uniform mat4 theMatrix;
uniform mat4 rotMatrix;
uniform vec3 light;
uniform int clock;

out vec2 vertexTexcoords;
out float intensity;

vec3 transnormal = normalize(rotMatrix * vec4(normal.x, normal.y, normal.z, 1)).xyz;
vec3 varPos;
vec3 varNorm;

vec2 Rotate2D(vec2 vec_in, float angle)
{
  vec2 vec_out;
  vec_out.x=cos(angle)*vec_in.x-sin(angle)*vec_in.y;
  vec_out.y=sin(angle)*vec_in.x+cos(angle)*vec_in.y;
  return vec_out;
}

void main() 
{
  varPos = position;
  varNorm = transnormal;
  varPos.xz = Rotate2D(varPos.xz, (varPos.y + 10)/10.0 * clock/100.0);
  varNorm.xz = Rotate2D(varNorm.xz, (varPos.y + 10)/10.0 * clock/100.0);
  intensity = dot(varNorm, normalize(light));
  gl_Position = theMatrix * vec4(varPos, 1);
   
  vertexTexcoords = vec2(texcoords.x + clock , texcoords.y);
}
"""

fragment_shader = """
#version 460
layout(location = 0) out vec4 fragColor;

uniform int clock;
in vec2 vertexTexcoords;
in float intensity;

uniform sampler2D tex;

void main()
{
  if (intensity > 0.05) {
    fragColor = texture(tex, vertexTexcoords) * intensity;
  } else {
    fragColor =  texture(tex, vertexTexcoords) * 0.05;
  }
}
"""

fragment_shader2 = """
#version 460
layout(location = 0) out vec4 fragColor;

uniform int clock;
in vec3 color;
in float intensity;

uniform sampler2D tex;

void main()
{
  fragColor =  vec4(color * intensity, 1.0);
}
"""

cvs1 = compileShader(vertex_shader, GL_VERTEX_SHADER)
cfs1 = compileShader(fragment_shader, GL_FRAGMENT_SHADER)
shader1 = compileProgram(cvs1, cfs1)

cvs2 = compileShader(vertex_shader2, GL_VERTEX_SHADER)
cfs2 = compileShader(fragment_shader2, GL_FRAGMENT_SHADER)
shader2 = compileProgram(cvs2, cfs2)

cvs3 = compileShader(vertex_shader3, GL_VERTEX_SHADER)
shader3 = compileProgram(cvs3, cfs1)

#Change shader here
shader = shader2

mesh = Obj('./models/jupiter.obj')
v_buffer = []

for face in mesh.faces:
  if len(face) == 3:
    for v in face:
      vertex = mesh.vertices[v[0]-1]
      normal = mesh.normals[v[2]-1]
      tvertex = mesh.tvertices[v[1]-1]
      v_buffer.extend([vertex, normal, tvertex])
  else:
    t = [[0,1,2],[0,2,3]]
    for i in t:
      for j in i:
        vertex = mesh.vertices[face[j][0]-1]
        normal = mesh.normals[face[j][2]-1]
        tvertex = mesh.tvertices[face[j][1]-1]
        v_buffer.extend([vertex, normal, tvertex])

face_count = len(mesh.faces)
vertex_data = numpy.array(v_buffer, dtype=numpy.float32).flatten()

vertex_buffer_object = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object)
glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

vertex_array_object = glGenVertexArrays(1)
glBindVertexArray(vertex_array_object)
glVertexAttribPointer(
  0, # location
  3, # size
  GL_FLOAT, # tipo
  GL_FALSE, # normalizados
  4 * 9, # stride
  ctypes.c_void_p(0)
)
glEnableVertexAttribArray(0)

glVertexAttribPointer(
  1, # location
  3, # size
  GL_FLOAT, # tipo
  GL_FALSE, # normalizados
  4 * 9, # stride
  ctypes.c_void_p(4 * 3)
)
glEnableVertexAttribArray(1)
glVertexAttribPointer(
  2, # location
  3, # size
  GL_FLOAT, # tipo
  GL_FALSE, # normalizados
  4 * 9, # stride
  ctypes.c_void_p(4 * 6)
)
glEnableVertexAttribArray(2)

glUseProgram(shader)

texture_surface = pygame.image.load('./models/jupiter.bmp')
texture_data = pygame.image.tostring(texture_surface, 'RGB', True)
texture_buffer = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture_buffer)
glTexImage2D(
  GL_TEXTURE_2D,
  0,
  GL_RGB,
  texture_surface.get_width(),
  texture_surface.get_height(),
  0,
  GL_RGB,
  GL_UNSIGNED_BYTE,
  texture_data
)
glGenerateMipmap(GL_TEXTURE_2D)

glUniform3f(glGetUniformLocation(shader, 'light'), -5, 0, 5)

def render(x, y, z, clock, rotx, roty, model_pos):
  i = glm.mat4(1)
  i

  translate = glm.translate(i, glm.vec3(model_pos[0], model_pos[1], -model_pos[2]))
  incline = glm.rotate(i, pi/12, glm.vec3(0, 0, 1))
  rotate = incline * glm.rotate(i, glm.radians(clock), glm.vec3(0, 1, 0))
  scale = glm.scale(i, glm.vec3(0.1, 0.1, 0.1))

  yunder = 5 * cos(roty)

  model = translate * rotate * scale
  view = glm.lookAt(
    glm.vec3(x, y, -z),
    glm.vec3(x + yunder * sin(rotx), y + 5 * sin(roty), -(z + yunder * cos(rotx))),
    glm.vec3(-sin(roty)*sin(rotx), cos(roty), sin(roty)*cos(rotx))
  )
  projection = glm.perspective(glm.radians(45), 1200/720, 0.1, 1000.0)

  theMatrix = projection * view * model

  glUniformMatrix4fv(
    glGetUniformLocation(shader, 'theMatrix'),
    1,
    GL_FALSE,
    glm.value_ptr(theMatrix)
  )
  glUniformMatrix4fv(
    glGetUniformLocation(shader, 'rotMatrix'),
    1,
    GL_FALSE,
    glm.value_ptr(rotate)
  )
  glUniform1i(glGetUniformLocation(shader, 'clock'), clock)

glViewport(0, 0, 1200, 720)

x = 0
y = 0
z = 0
a = 1/20
t = 0
model_radius = 1
model_pos = (0, 0, 5)
rotx = 0
roty = 0
running = True
while running:
  t += 1
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

  render(x, y, z, t, rotx, roty, model_pos)
 
  glDrawArrays(GL_TRIANGLES, 0, face_count * 6)

  keys = pygame.key.get_pressed()
  
  if keys[pygame.K_a]:
    if (x - a * cos(rotx) - model_pos[0])**2 + (y - model_pos[1])**2 + (z + a * sin(rotx) - model_pos[2])**2 > model_radius**2:
      x -= a * cos(rotx)
      z += a * sin(rotx)
  if keys[pygame.K_d]:
    if (x + a * cos(rotx) - model_pos[0])**2 + (y - model_pos[1])**2 + (z - a * sin(rotx) - model_pos[2])**2 > model_radius**2:
      x += a * cos(rotx)
      z -= a * sin(rotx)
  if keys[pygame.K_s]:
    if (x - a * sin(rotx) - model_pos[0])**2 + (y - model_pos[1])**2 + (z - a * cos(rotx) - model_pos[2])**2 > model_radius**2:
      z -= a * cos(rotx)
      x -= a * sin(rotx)
  if keys[pygame.K_w]:
    if (x + a * sin(rotx) - model_pos[0])**2 + (y - model_pos[1])**2 + (z + a * cos(rotx) - model_pos[2])**2 > model_radius**2:
      z += a * cos(rotx)
      x += a * sin(rotx)
  if keys[pygame.K_LEFT]:
    rotx -= a/2
  if keys[pygame.K_RIGHT]:
    rotx += a/2
  if keys[pygame.K_DOWN]:
    roty -= a/2 if roty > -(pi/2 - a/2) else 0
  if keys[pygame.K_UP]:
    roty += a/2 if roty < (pi/2 - a/2) else 0
  if keys[pygame.K_SPACE]:
    if (x - model_pos[0])**2 + (y + a - model_pos[1])**2 + (z - model_pos[2])**2 > model_radius**2:
      y += a
  if keys[pygame.K_LSHIFT]:
    if (x - model_pos[0])**2 + (y - a - model_pos[1])**2 + (z - model_pos[2])**2 > model_radius**2:
      y -= a

  pygame.display.flip()
  clock.tick(60)

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      running = False