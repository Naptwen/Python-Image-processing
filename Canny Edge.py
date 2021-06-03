import sys
import numpy as np
import math
def Canny(self):
    self.Gray()
    self.Histo()
    self.Gauss()
    self.Sobel()
    self.Edge_detect2()
    self.Show()
    
def Gray(self):
    for i in range(len(self.image)):
        for j in range(len(self.image[i])):
            R = self.image[i,j,0]
            G = self.image[i,j,1]
            B = self.image[i,j,2]
            W = 0.299*R + 0.587*G + 0.114*B
            self.temp_image[i,j] = W
        
def Histo(self):
    rows = len(self.temp_image)
    cols = len(self.temp_image[0])
    size = 256
    #------histogram----------
    histogram = np.zeros(size)
    for i in range(len(self.temp_image)):
        for j in range(len(self.temp_image[i])):
             index = int(self.temp_image[i,j])
             histogram[index] += 1
    #------histogram_equalization-----------
    total_histogram = histogram.sum()
    equalization_histogram = np.zeros(size)
    cdf = 0
    for i in range(size):
        pn = histogram[i]/total_histogram
        cdf += pn
        equalization_histogram[i] = (size - 1) * cdf
    #----convert data----------------------
    for i in range(len(self.temp_image)):
        for j in range(len(self.temp_image[i])):
            self.temp_image[i,j] = equalization_histogram[int(self.temp_image[i,j])]
            
def Sobel(self):
    total_val = 0
    temp = self.temp_image
    for i in range(len(self.temp_image)-3):
        for j in range(len(self.temp_image[i])-3):
            V = [[-1, 0, 1],[ -2, 0, 2],[ -1, 0, 1]]
            H = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
            mat = self.temp_image[i:i+3,j:j+3]
            Gx = np.dot(V,mat).sum()
            Gy = np.dot(H,mat).sum()
            temp[i , j] = math.sqrt(Gx*Gx + Gy*Gy)
    self.temp_image = copy.deepcopy(temp)
    
def Gauss(self):
    size = 2 #size 2 gaussian filter
    temp = copy.deepcopy(self.temp_image)
    rows  = len(self.temp_image)
    cols = len(self.temp_image[0])
    self.gauss_dir = np.zeros((rows, cols))
    for i in range(len(temp)-5):
        for j in range(len(temp[i])-5):
            x, y = np.mgrid[-size : size + 1, -size : size + 1]
            normal = 1/ (2 * np.pi * 1.4 * 2)
            g = np.exp(-(x**2 + y**2)/(2 * 1.4**2)) * normal
            g = g/g.sum()
            temp[i : i + 5, j : j + 5] = np.dot(self.temp_image[i : i + 5, j : j + 5], g)
    self.temp_image = np.clip(temp, 0, 255)

def Edge_detect2(self): #Applying A* algorithm for edge detecting
    max = 30
    min = 2
    weak_list = np.zeros((len(self.temp_image),len(self.temp_image[0]),2))
    open_list = np.empty((0,2), int)
    for i in range(len(self.temp_image)):
        for j in range(len(self.temp_image[i])):
            if self.temp_image[i,j] > max:
                self.temp_image[i,j] = 255
            elif self.temp_image[i,j] < min:
                self.temp_image[i,j] = 0
            else:
                weak_list[i,j] = [-1, -1];
                open_list = np.vstack([open_list, [i, j]]);
    closed_list = np.empty((0,2), int)
    while len(open_list) > 0:
        a = open_list[0,0]
        b = open_list[0,1]
        open_list = np.delete(open_list, (0), axis=0)
        if weak_list[a,b,0] != -2 :
            for i in range(-1,1):
                for j in range(-1,1):
                    x = a + i
                    y = b + j
                    if x > -1 and x < len(self.temp_image) and y > -1 and  y <= len(self.temp_image[0]):
                        if self.temp_image[x,y] > max:
                            u = a
                            v = b
                            while True:
                                closed_list = np.vstack([closed_list, [u, v]])
                                gx = weak_list[u,v,0]
                                gy = weak_list[u,v,1]
                                weak_list[u,v,0] = -2
                                if gx == -1 or gx == -2:
                                    break
                                else:
                                    u = weak_list[gx, gy, 0]
                                    v = weak_list[gx, gy, 1]
                        elif self.temp_image[x,y] > min:
                            if weak_list[x,y,0] == -1:
                                weak_list[x,y,0] = a
                                weak_list[x,y,1] = b
    for i in range(len(closed_list)):
        x = closed_list[i,0]
        y = closed_list[i,1]
        self.temp_image[x,y] = 255
        
    for i in range(len(self.temp_image)):
        for j in range(len(self.temp_image[i])):
            if self.temp_image[i,j] != 255:
                self.temp_image[i,j] = 0
                        
        

def Edge_detect(self):
    max = 100
    min = 20
    weak_list = np.empty((0,2), int)
    for i in range(len(self.temp_image)):
        for j in range(len(self.temp_image[i])):
            if self.temp_image[i,j] > max:
                self.temp_image[i,j] = 255
            elif self.temp_image[i,j] < min:
                self.temp_image[i,j] = 0
            else:
                weak_list = np.vstack([weak_list, [i, j]]);
    print(weak_list)
    closed_list = np.empty((0,2), int)
    trial = 0
    while len(weak_list) > 0 and trial < 10000:
        trial += 1
        start = weak_list[0]
        closed_list = np.vstack([closed_list, start])
        #delete from weaklist
        weak_list = np.delete(weak_list, (0), axis=0)
        connection = False;
        #8 direction checking
        for i in range(-1,1):
            for j in range(-1,1):
                #get index
                x = start[0] + i
                y = start[1] + j
                #check it is in matrix
                if x > -1 and x < len(self.temp_image) and y > -1 and  y <= len(self.temp_image[0]):
                    #if one of them is strong color pixel
                    if self.temp_image[x,y] > max:
                        #make the elements in closed list as the strong color pixel
                        for t in range(len(closed_list)):
                            a = closed_list[t,0]
                            b = closed_list[t,1]
                            self.temp_image[a,b] = 255
                        #empty array
                        closed_list = np.empty((0,2), int)
                        connection = True
                        break
                    #if negihbor is weak color pixel
                    if  self.temp_image[x,y] < max and self.temp_image[x,y] > min:
                        closed_list = np.vstack([closed_list, [x,y]])
                        #delete from weaklist
                        for u in range(len(weak_list)):
                            if (weak_list[u] == [x,y]).all():
                                weak_list = np.delete(weak_list, (u), axis=0)
                                break
                        connection = True
                        break
        if connection == False:
            for g in range(len(closed_list)):
                a = closed_list[g,0]
                b = closed_list[g,1]
                self.temp_image[a,b] = 0
                    
def Normal(self):
    max = self.temp_image[0,0]
    min = self.temp_image[0,0]
    for i in range(len(self.temp_image)):
        for j in range(len(self.temp_image[i])):
            if max < self.temp_image[i,j]:
                max = self.temp_image[i,j]
            if min > self.temp_image[i,j]:
                min = self.temp_image[i,j]
    for i in range(len(self.temp_image)):
        for j in range(len(self.temp_image[i])):
            self.temp_image[i,j] = (self.temp_image[i,j] - min) / (max-min) * 255
    self.Edge_detect()
 
