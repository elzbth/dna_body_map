#!/usr/bin/python

import random
import numpy
import Image
import sys
import time

from __init__ import * #LOAD DEPENDENCIES FROM FILE __init__.py
from display import Display
from viewPorts import *
from texture import Texture
from math import * # trigonometria

SCREEN_SIZE = [1024, 768]

def png_image_file_write(filename, number, data, size):
  image = Image.frombuffer("RGBA", size, data, "raw", "RGBA", 0,0)
  znumber = "%05d"%number
  image.save(filename + znumber + ".png")

def createAndCompileShader(type, sourceLike):
  shader=glCreateShader(type)
  glShaderSource(shader,sourceLike)
  glCompileShader(shader)

  result=glGetShaderiv(shader,GL_COMPILE_STATUS)
  if (result!=1):
    raise Exception("Imposs shader:\n"+glGetShaderInfoLog(shader))
  return shader

class Perspective:
      """Perspective class"""
      def __init__(self):
        pygame.init()
        self.display = Display(SCREEN_SIZE)
        self.frames_per_second = pygame.time.Clock()
        self.done = False
        self.t = 0

        self.vertex_shader= createAndCompileShader(GL_VERTEX_SHADER,"""
varying vec3 normal, lightDir;
void main()
{
normal = gl_NormalMatrix * gl_Normal;
vec4 posEye = gl_ModelViewMatrix * gl_Vertex;

gl_TexCoord[0] = gl_TextureMatrix[0]*gl_ModelViewMatrix*gl_Vertex;
// LightSource[0] position is assumed to be the projector position

lightDir = vec3(gl_LightSource[0].position.xyz - posEye.xyz);
gl_Position = gl_ProjectionMatrix * gl_ModelViewMatrix * gl_Vertex;

}"""
        );

        self.fragment_shader=createAndCompileShader(GL_FRAGMENT_SHADER,"""
uniform sampler2D projMap;
varying vec3 normal, lightDir;

void main (void)
{
vec4 final_color = vec4(0.0,0.5,0,0.3);
vec3 N = normalize(normal);
vec3 L = normalize(lightDir);
float lambert = dot(N,L);
if( gl_TexCoord[0].z>0.0 // in front of projector?
&&
lambert>0 ) // facing projector?
{
vec4 ProjMapColor = texture2DProj(projMap, gl_TexCoord[0].xyz);
final_color += ProjMapColor*lambert*pow(length(L),-2.0);
}
gl_FragColor = final_color;
}
""");

        self.program=glCreateProgram()
        glAttachShader(self.program,self.vertex_shader)
        glAttachShader(self.program,self.fragment_shader)
        glLinkProgram(self.program)

        try:
          glUseProgram(self.program)
        except OpenGL.error.GLError:
          print glGetProgramInfoLog(self.program)
          raise
        glNewList(1,GL_COMPILE)
        glBegin(GL_QUADS)
        glColor3f(1,1,1)

        glNormal3f(0,0,-1)
        glVertex3f( -1, -1, -1)
        glVertex3f( 1, -1, -1)
        glVertex3f( 1, 1, -1)
        glVertex3f( -1, 1, -1)

        glNormal3f(0,0,1)
        glVertex3f( -1, -1, 1)
        glVertex3f( 1, -1, 1)
        glVertex3f( 1, 1, 1)
        glVertex3f( -1, 1, 1)

        glNormal3f(0,-1,0)
        glVertex3f( -1, -1, -1)
        glVertex3f( 1, -1, -1)
        glVertex3f( 1, -1, 1)
        glVertex3f( -1, -1, 1)

        glNormal3f(0,1,0)
        glVertex3f( -1, 1, -1)
        glVertex3f( 1, 1, -1)
        glVertex3f( 1, 1, 1)
        glVertex3f( -1, 1, 1)

        glNormal3f(-1,0,0)
        glVertex3f( -1, -1, -1)
        glVertex3f( -1, 1, -1)
        glVertex3f( -1, 1, 1)
        glVertex3f( -1, -1, 1)

        glNormal3f(1,0,0)
        glVertex3f( 1, -1, -1)
        glVertex3f( 1, 1, -1)
        glVertex3f( 1, 1, 1)
        glVertex3f( 1, -1, 1)

        glEnd()
        glEndList()

        self.texture=glGenTextures(1)
        glActiveTexture(GL_TEXTURE0);
        glBindTexture( GL_TEXTURE_2D, self.texture );
        glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );

        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)

        self.texdata = numpy.zeros((256,256,4))

        for i in range(0,256):
          x=(i-128.1)/128.0
          for j in range(0,256):
            y = (j-128.1)/128.0
            if ((x*x+y*y>0.9) | (x*x+y*y<0.3)):
              self.texdata[i][j][0] = 0
              self.texdata[i][j][1] = 0
              self.texdata[i][j][2] = 0
              self.texdata[i][j][3] = 0
            else:
              self.texdata[i][j][0] = 1
              self.texdata[i][j][1] = 1
              self.texdata[i][j][2] = 1
              self.texdata[i][j][3] = 1
        glTexImage2Df(GL_TEXTURE_2D, 0,GL_RGBA,0,GL_RGBA,self.texdata)
        self.loc = glGetUniformLocation(self.program,"projMap");
        glUniform1i(self.loc, 0)
        glEnable(GL_DEPTH_TEST)

        while True:
          self.t += 1

          ppos=[sin(self.t/260.0)*4,cos(self.t/240.0)*4,0]
          palpha=self.t
          pbeta=self.t/3.0

          glMatrixMode(GL_TEXTURE);
          glLoadIdentity()
          glRotate(-palpha,0,1,0);
          glRotate(-pbeta ,0,0,1);
          glTranslate(-ppos[0],-ppos[1],-ppos[2])

          glLightfv(GL_LIGHT0,GL_POSITION,[ppos[0],ppos[1],ppos[2]]);

          glMatrixMode(GL_PROJECTION)
          glLoadIdentity()
          gluPerspective(90,1,0.01,1000)
          gluLookAt(sin(self.t/200.0)*8,sin(self.t/500.0)*3+8,cos(self.t/200.0)*8,0,0,0,0,1,0)

          glClearColor(0.0, 0.0, 0.0, 1.0)
          glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
          glMatrixMode(GL_MODELVIEW)

          glLoadIdentity()
          glTranslate(ppos[0],ppos[1],ppos[2])
          glBegin(GL_TRIANGLES)

          glVertex3f(1,0,0)
          glVertex3f(0,1,0)
          glVertex3f(0,0,1)

          glEnd()

          glColor3f(1,1,1)
          glLoadIdentity()

          for i in range(-1,2):
            for j in range(-1,2):
              for k in range(-1,2):
                glPushMatrix()
                glTranslate(i*5,j*5,k*5)
                glScale(1,1,1)
                glCallList(1)
                glPopMatrix()
          time.sleep(0.01);
          self.display.get_events()
          self.display.set_flip_option()

if __name__ == '__main__':
  Perspective()