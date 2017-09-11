#!/usr/bin/env python

'''
Seguimiento Lucas-Kanade
====================

Buscaremos dar seguimiento a puntos interesantes de la imagen utilizando la funcion goodFeaturesToTrack


Uso
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2
import video
from common import anorm2, draw_str
from time import clock

"""
Primero definimos los criterios para el seguimiento
winzise mide el tamano de la ventana en el que se analizara el movimiento.
maxLevel mide el numero de niveles de piramide que se usaran, iniciando de 0
criteria es un conjunto de parametros:
    criteria.maxcount cuenta el numero maximo de puntos interesantes a seguir.
    criteria.epsilon dice cuanto puede moverse algo para ser considerado el mismo punto.

"""
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

"""

Aqui definimos los criterios para Shi-Tomasi
"""

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )



class App:

    
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = video.create_capture(video_src) #iniciamos la captura del video
        self.frame_idx = 0 #frame index

    def run(self):

        color=np.random.randint(0,255,(100,3))
        ret, old_frame=self.cam.read()
        mask=np.zeros_like(old_frame)
        old_gray=cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
        p0=cv2.goodFeaturesToTrack(old_gray,mask=None, **feature_params)
        while True:
            _ret, frame = self.cam.read()
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            p1,st,err=cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            good_new = p1[st==1]
            good_old = p1[st==1]

            for i, (new,old) in enumerate(zip(good_new, good_old)):
                a,b=new.ravel()
                c,d=old.ravel()
                mask= cv2.line(mask, (a,b), (c,d), color[i].tolist(),2)
                frame=cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
            img = cv2.add(frame,mask)
            
            cv2.imshow('frame', img)

            ch = cv2.waitKey(1)
            if ch == 27:
                break

            old_gray= frame_gray.copy()
            p0=good_new.reshape(-1,1,2)

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
