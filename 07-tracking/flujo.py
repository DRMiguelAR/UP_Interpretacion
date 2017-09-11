#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock

'''
Flujo optico denso
====================

Encontraremos el flujo optico para cada punto de la imagen.
Este algoritmo esta basado en el trabajo de Gunner Farneback (2013)

Se encuentra un arreglo de dos canales: u y v.
Para cada uno se encuentra magnitud y direccion. Se guarda en una imagen:
    Direccion: Hue
    Magnitud: value

Uso:
-----
flujo.py [<video_source>]


Teclas:
----
ESC - exit
'''


class App:

    
    def __init__(self, video_src):
        self.cap = video.create_capture(video_src) #iniciamos la captura del video

    def run(self):
        ret, frame1 = self.cap.read()
        prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
        while True:
            ret, frame2 = self.cap.read()
            next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2',rgb)
            
            ch = cv2.waitKey(1)
            if ch == 27:
                break
            elif ch==ord('s'):
                cv2.imwrite("flujoOptico.png",frame2)
                cv2.imwrite("hsvOptico.png",rgb)
            prvs=next 
            
def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
