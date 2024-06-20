from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, patches
import collections
from collections import Counter
from scipy import optimize

#Free space path loss
freq=5250e6     # in Hz - Channel 50 wifi
c=3e8           # speed of light in vaccum in m/s
Pt= 20          # power transmitted dBm - set as the maximum for Wi-Fi
noise=-85       # noise floor -85 dBm
step=1          # in the points search
maxMCS=780      # capacity of the shared wireless medium Mbits/s

SNR_MARGIN = 1

#Variables to calculate energy consumption
rho=1.225
W=20
R=0.4
A=0.503
omega=300
Utip=120
d0=0.6
k=0.1
V0=4.03
delta=0.012
s=0.05
g=9.8 #gravitacional force aproximation

P0=(delta/8)*rho*s*A*math.pow(omega,3)*math.pow(R,3)
Pi=(1+k)*(math.pow(W,3/2)/math.sqrt(2*rho*A))

alt_min=6
flag=0

colors = ['b','r','g','m','y','c','k','b','r']

#Track Total Energy Hour Consumption
TEH_Trajectory = 0
TEH_Hovering = 0
TEH_Optimal_Rotary= 0
TEH_Optimal_Fixed= 0  
TEH_Circular = 0
TEH_Oval_Circular = 0
TEH_Oval_Area = 0
TEH_SUPPLY_ROTARY = 0
TEH_SUPPLY_FIXED = 0
TEH_SUPPLY_Aux = 0
Best_flag=0

# Rotary-Wing Energy Consumption Model-----------------------------------
def P_rotary(V,r):
    firstElement= P0*(1+(3*math.pow(V,2)/(math.pow(Utip,2))))
    square=1+math.pow(math.pow(V,2)/r,2)/math.pow(g,2)+(math.pow(V,4)/(4*math.pow(V0,4)))
    secondElement=Pi*math.sqrt(1+math.pow(math.pow(V,2)/r,2)/math.pow(g,2))*math.pow((math.sqrt(square)-(math.pow(V,2)/(2*math.pow(V0,2)))),1/2)
    thirdElement=(1/2)*d0*rho*s*A*math.pow(V,3)
    return firstElement+secondElement+thirdElement
#-----------------------------------------------------------------------

# Fixed-Wing Energy Consumption Model------------------------------------
c1 = 0.000926
c2 = 2250

def P_fixed(V,r):
    return (c1+c2/(((math.pow(g,2))*(math.pow(r,2)))))*math.pow(V,3)+c2/V
#-----------------------------------------------------------------------

# Radius for which the power recieved is equal or greater than the desired
def distanceForSNR(SNR):
    exponent= (-SNR-noise+Pt+20*math.log10(c/(4*freq*math.pi)))/20
    return  math.pow(10, exponent) #radius for which the power recieved is equal or greater than the desired

def Euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2-x1)**2 + (y2-y1)**2)

# Function to return the minimum distance between a line segment AB and a point E https://www.geeksforgeeks.org/minimum-distance-from-a-point-to-the-line-segment-using-vectors/
def minDistance(A, B, E) :
    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]
 
    # vector BP
    BE = [None, None]
    BE[0] = E[0] - B[0]
    BE[1] = E[1] - B[1]
 
    # vector AP
    AE = [None, None]
    AE[0] = E[0] - A[0]
    AE[1] = E[1] - A[1]
 
    # Variables to store dot product
 
    # Calculating the dot product
    AB_BE = AB[0] * BE[0] + AB[1] * BE[1]
    AB_AE = AB[0] * AE[0] + AB[1] * AE[1]
 
    # Minimum distance from
    # point E to the line segment
    reqAns = 0
 
    # Case 1
    if (AB_BE > 0) :
        # Finding the magnitude
        y = E[1] - B[1]
        x = E[0] - B[0]
        reqAns = math.sqrt(x * x + y * y)
 
    # Case 2
    elif (AB_AE < 0) :
        y = E[1] - A[1]
        x = E[0] - A[0]
        reqAns = math.sqrt(x * x + y * y)
 
    # Case 3
    else:
        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AE[0]
        y2 = AE[1]
        mod = math.sqrt(x1 * x1 + y1 * y1)
        reqAns = abs(x1 * y2 - y1 * x2) / mod
     
    return reqAns

#Open and read file with GUs information
f=open("GUs.txt", "r")

circular = open('circular.txt', 'w')
oval_c = open('oval_c.txt', 'w')
oval_a = open('oval_a.txt', 'w')
circular_fixed = open('circular_fixed.txt', 'w')
oval_c_fixed = open('oval_c_fixed.txt', 'w')
oval_a_fixed = open('oval_a_fixed.txt', 'w')

f.readline() #read number of groups line
nGroups=f.readline().strip()
print("nGroups: " + str(nGroups))
f.readline() #read numbers line
nGUs=int(f.readline())
print("nGUs: " + str(nGUs))
f.readline() #read positions line
line=f.readline().strip().split(",")

n_points=[]
nGUs_group=[]   #Stores number of GUs per group
TotalPoints=[]  #Stores the points of all intersetion areas

#Dictionary for SNR and Data rate relation
dicMCS=[
    {"SNR":13.1,"data_rate":53/nGUs},
    {"SNR":13.6,"data_rate":103/nGUs},
    {"SNR":16.1,"data_rate":152/nGUs},
    {"SNR":19.5,"data_rate":198/nGUs},
    {"SNR":22.6,"data_rate":287/nGUs},
    {"SNR":27.1,"data_rate":368/nGUs},
    {"SNR":28.4,"data_rate":405/nGUs},
    {"SNR":29.9,"data_rate":447/nGUs},
    {"SNR":34.1,"data_rate":518/nGUs},
    {"SNR":35.3,"data_rate":553/nGUs}
]


for i in range(int(nGroups)):
    nGUs_group.append(int(line[i]))

print("nGUs_group: " + str(nGUs_group) + "\n")

for n in range(int(nGroups)):
    GUs=[]
    x=[]
    y=[]
    z=[]
    traffic=[]
    f.readline()
    for j in range (nGUs_group[n]):
        GUs.append((f.readline().split(",")))
        x.append(float(GUs[j][0]))
        y.append(float(GUs[j][1]))
        z.append(float(GUs[j][2]))
        traffic.append(float(GUs[j][3]))
        if(traffic[j]>(maxMCS/nGUs)):
            traffic[j]=maxMCS/nGUs
        j+=1
        

    print(x)
    print(y)
    print(z)
    print(traffic)
    
    #Map traffic to SNR
    data_rate_val=[None]*len(traffic)

    j=0
    while j < len(traffic):
        i=0
        sums=[]
        while i<10:
            current=dicMCS[i].get("data_rate")-traffic[j]
            if(current>=0):
                sums.append(current)
                if(min(sums)==current):
                    data_rate_val[j]=dicMCS[i].get("data_rate")
            i+=1  
        j+=1

    SNR_values=[]
    for j in data_rate_val:
        i=0
        while i<10:
            data_rate=dicMCS[i].get("data_rate")
            if(j==data_rate):
                SNR_values.append(dicMCS[i].get("SNR") + SNR_MARGIN)
            i+=1

    xToCalcPos=[]*len(SNR_values)
    xToCalcNeg=[]*len(SNR_values)
    yToCalcPos=[]*len(SNR_values)
    yToCalcNeg=[]*len(SNR_values)
    zToCalcPos=[]*len(SNR_values)
    zToCalcNeg=[]*len(SNR_values)

    j=0
    for i in SNR_values:
        dist=distanceForSNR(i)
        xToCalcPos.append(x[j]+math.floor(dist))
        xToCalcNeg.append(x[j]-math.floor(dist))
        yToCalcPos.append(y[j]+math.floor(dist))
        yToCalcNeg.append(y[j]-math.floor(dist))
        zToCalcPos.append(z[j]+math.floor(dist))
        zToCalcNeg.append(z[j]-math.floor(dist))
        j+=1

    #min/max positions
    print("\nMin/Max Pos:")
    print(xToCalcPos)
    print(xToCalcNeg)
    print(yToCalcPos)
    print(yToCalcNeg)
    print(zToCalcPos)
    print(zToCalcNeg)
    print("\n")
 
    '''#figure for positions

    fig= plt.figure()
    ax = plt.axes(projection='3d')
    #ax.set_title('GUs Position Group %d' %(n+1))
    ax.set_title('GUs Position and Spheres', fontsize=22)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    #Generate GUs

    i=0
    while i < len(x):
        ax.scatter(x[i],y[i],z[i],marker='o', label="GU"+str(i+1))
        i+=1

    #Generate spheres
    i=0
    xs=[None]*int(nGUs)
    ys=[None]*int(nGUs)
    zs=[None]*int(nGUs)
    while i<len(x):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        distance=distanceForSNR(SNR_values[i])
        print("Distance: ", distance)
        xs[i] = x[i] + distance * np.outer(np.cos(u), np.sin(v))
        ys[i] = y[i] + distance* np.outer(np.sin(u), np.sin(v))
        zs[i] = z[i] + distance * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs[i], ys[i], zs[i],  rstride=4, cstride=4,alpha=0.5)
        ax.set_xlim(0,100)
        ax.set_ylim(0,100)
        ax.set_aspect('equal', adjustable='box')

        i+=1 '''

    # Calculate points where all of the SNR is >= threshold
    i=0

    pd= [[]]*len(x)

    while i<len(x):
        pd[i]= np.array((x[i],y[i],z[i]))
        i+=1

    # Beginning of PREP
    xmax , ymax, zmax = max(xToCalcPos), math.floor(max(yToCalcPos)), alt_min   #max(zToCalcPos) assim só tem de verificar uma altura
    xd, yd, zd = min(xToCalcNeg), min(yToCalcNeg), alt_min                      #zd=altura minima se zd=min(zToCalcNeg) fica definida pelas esferas

    def calculateValidPoints(pd,xmax,ymax,zmax,xd,yd,zd,SNR_values):
        validPoints = []
        dist = [None]*len(x)
        while xd <= xmax:
            yd=min(yToCalcNeg)
            count=0
            while yd <= ymax:
                zd=alt_min #zd=altura minima se zd=min(zToCalcNeg) fica definida pelas esferas
                count=0
                while zd <= zmax:
                    currentPoint=np.array((xd,yd,zd))
                    #print('Current Point ='+str(currentPoint))
                    i=0
                    count=0
                    while i<len(x):
                        dist[i] = np.linalg.norm(pd[i]-currentPoint)
                        if(dist[i]==0):
                            Pr=Pt
                            if((Pr-noise)>=SNR_values[i]):
                                count+=1
                        elif(dist[i]>0.0):
                            Pr=Pt+20*math.log10(c/(4*freq*dist[i]*math.pi))
                            if((Pr-noise)>=SNR_values[i]):
                                count+=1
                        #print("FMAP"+str(i)+" with SNR: "+str(SNR_values[i]))        
                        dist[i]=None
                        i+=1    
                    if(count==len(x)):
                        validPoints.append(currentPoint)
                    zd+=step
                yd+=step
            xd+=step

        return validPoints

    happened=0

    validPoints=calculateValidPoints(pd,xmax,ymax,zmax,xd,yd,zd,SNR_values)
    
    if(len(validPoints)==0):
        print("No intersection was found")
        exit()

    print("Length: ",len(validPoints))
    print("SNR values = ",SNR_values)
    print("Transmit Power: ",Pt)

    #Plot for the points for the volume admissible
    validAltitudes=[]

    for j in validPoints:
        #ax.scatter(j[0],j[1],j[2],marker='o')
        validAltitudes.append(j[2])

    occurences= Counter(validAltitudes) # Find the altitude with the biggest area
    #print(occurences)
    desiredAltitude= occurences.most_common(1)[0][0] 

    print("Desired Altitude= "+str(desiredAltitude))

    if(happened!=0):
        fig = plt.figure()

        ax = plt.axes(projection='3d')
        ax.set_title('SNR for FMAPs')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        i=0
        while i<len(x):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)

            distance=distanceForSNR(SNR_values[i])
            xs = x[i] + distance * np.outer(np.cos(u), np.sin(v))
            ys = y[i] + distance* np.outer(np.sin(u), np.sin(v))
            zs = z[i] + distance * np.outer(np.ones(np.size(u)), np.cos(v))
            ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, alpha=0.5)
            i+=1

    #plot area for desired altitude
    '''fig = plt.figure()

    ax = plt.axes(projection='3d')
    #ax.set_title('Intersection Area Group %d' %(n+1))
    ax.set_title('Intersection Area', fontsize=22)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    '''
    
    pointsArea=[] #Points in the desired altitude

    for j in validPoints:
        if(j[2]==desiredAltitude):
            #ax.scatter(j[0],j[1],j[2], marker='o', color=colors[n]) 
            pointsArea.append(j)
    #ax.set_xlim(0,100)
    #ax.set_ylim(0,100)
    
    # Verify for area collision -> removes points from new area if they are already in another

    '''ax.set_box_aspect([50,50,15])
    ax.set_aspect('equal', adjustable='box')'''

    j=0
    if TotalPoints!=[]: 
        for ind, j in enumerate(pointsArea):
            for l in TotalPoints:
                if (np.array_equal(j, l)):
                    flag=1
                    print(j,l)
                    pointsArea.pop(ind)


    TotalPoints.extend(pointsArea)
    print("Points in area: ", len(pointsArea))
    n_points.append(len(TotalPoints))
    
    #Find the centroid for the ideal position
    xarray=[j[0] for j in pointsArea]
    yarray=[j[1] for j in pointsArea]

    idealPos=[sum(xarray)/len(pointsArea),sum(yarray)/len(pointsArea),desiredAltitude]
    idealPosnoZ=[sum(xarray)/len(pointsArea),sum(yarray)/len(pointsArea)]

    print("Ideal Position= "+str(idealPos))

    
    '''fig = plt.figure()

    ax = plt.axes(projection='3d')
    #ax.set_title('Perimeter and Circular Trajectory Group %d' %(n+1))
    ax.set_title('Circular', fontsize = 22)
    #ax.set_title('EREP')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')'''
    
    #Find perimeter points
    minX=min(xarray)
    maxX=max(xarray)
    perimeter=[]

    X=minX
    while(X<=maxX):
        perimeter_aux=[]
        for j in pointsArea:
            if j[0]==X:
                perimeter_aux.append(j) 

        if perimeter_aux!=[]:
            minY=min(j[1] for j in perimeter_aux)
            maxY=max(j[1] for j in perimeter_aux)

            for p in perimeter_aux:
                if((p[1]==minY or p[1]==maxY) or p[0]==minX or p[0]==maxX):
                    #ax.scatter(p[0],p[1],p[2],marker='o', color=colors[n]) 
                    perimeter.append(p)
        X+=1
    
    distances=[]

    #-----------------------------------------------------------
    #Define radius -> minimum distance from the ideal position to the perimeter
    for j in perimeter:
        distances.append(Euclidean_distance(j[0], j[1], idealPos[0], idealPos[1]))

    r = min(distances)

    if (maxX-minX)/2 < r:
       r=(maxX-minX)/2

    if len(perimeter) <= 2:
        r=0
    #-----------------------------------------------------------

    #define points
    ymin=min(yarray)
    ymax=max(yarray)
    xmax=max(xarray)
    xmin=min(xarray)

    xymin=[] 
    xymax=[] 
    yxmin=[] 
    yxmax=[] 

    for j in pointsArea:
        if(j[0]==xmin):
            yxmin.append(j[1])
        if(j[0]==xmax):
            yxmax.append(j[1])
        if(j[1]==ymin):
            xymin.append(j[0])
        if(j[1]==ymax):
            xymax.append(j[0])

    def calculateDistanceT(point1,point2,point3,point4,pointI):
        pn1=np.array(point1)
        pn2=np.array(point2)
        pn3=np.array(point3)
        pn4=np.array(point4)
        pnI=np.array(pointI)

        d1=np.linalg.norm(pnI-pn1)
        d2=np.linalg.norm(pnI-pn2)
        d3=np.linalg.norm(pnI-pn3)
        d4=np.linalg.norm(pnI-pn4)
        d12=np.linalg.norm(pn2-pn1)
        d34=np.linalg.norm(pn4-pn3)

        dTotal=d1+d2+d3+d4+d12+d34

        return dTotal

    l1x=[0]*3
    l1y=[0]*3
    l2x=[0]*3
    l2y=[0]*3
    l3x=[0]*3
    l3y=[0]*3
    l4x=[0]*3
    l4y=[0]*3
    l5x=[0]*3
    l5y=[0]*3
    l6x=[0]*3
    l6y=[0]*3

    #trajectory 1 points

    point1T1=[min(xymin),ymin,desiredAltitude]
    point2T1=[max(xymin),ymin,desiredAltitude]
    point3T1=[min(xymax),ymax,desiredAltitude]
    point4T1=[max(xymax),ymax,desiredAltitude]

    """PLOT!!! 
    fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') """

    #define lines 
    lz=[desiredAltitude,desiredAltitude]
    #between p1 and p2
    l1x[0]=[point1T1[0],point2T1[0]]
    l1y[0]=[point1T1[1],point2T1[1]]
    #between p3 and p4
    l2x[0]=[point3T1[0],point4T1[0]]
    l2y[0]=[point3T1[1],point4T1[1]]
    #between p1 and ideal
    l3x[0]=[point1T1[0],idealPos[0]]
    l3y[0]=[point1T1[1],idealPos[1]]
    #between p2 and ideal
    l4x[0]=[point2T1[0],idealPos[0]]
    l4y[0]=[point2T1[1],idealPos[1]]
    #between p3 and ideal
    l5x[0]=[point3T1[0],idealPos[0]]
    l5y[0]=[point3T1[1],idealPos[1]]
    #between p4 and ideal
    l6x[0]=[point4T1[0],idealPos[0]]
    l6y[0]=[point4T1[1],idealPos[1]]
    #plot all of them
    """ ax.plot(l1x[0],l1y[0],lz)
    ax.plot(l2x[0],l2y[0],lz)
    ax.plot(l3x[0],l3y[0],lz)
    ax.plot(l4x[0],l4y[0],lz)
    ax.plot(l5x[0],l5y[0],lz)
    ax.plot(l6x[0],l6y[0],lz)
    ax.legend() """
    distanceT1=calculateDistanceT(point1T1,point2T1,point3T1,point4T1,idealPos)

    #trajectory 2

    """PLOT!!!
    fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') """


    point1T2=[xmin,min(yxmin),desiredAltitude]
    point2T2=[xmin,max(yxmin),desiredAltitude]
    point3T2=[xmax,min(yxmax),desiredAltitude]
    point4T2=[xmax,max(yxmax),desiredAltitude]

    distanceT2=calculateDistanceT(point1T2,point2T2,point3T2,point4T2,idealPos)

    #define lines 
    lz=[desiredAltitude,desiredAltitude]
    #between p1 and p2
    l1x[1]=[point1T2[0],point2T2[0]]
    l1y[1]=[point1T2[1],point2T2[1]]
    #between p3 and p4
    l2x[1]=[point3T2[0],point4T2[0]]
    l2y[1]=[point3T2[1],point4T2[1]]
    #between p1 and ideal
    l3x[1]=[point1T2[0],idealPos[0]]
    l3y[1]=[point1T2[1],idealPos[1]]
    #between p2 and ideal
    l4x[1]=[point2T2[0],idealPos[0]]
    l4y[1]=[point2T2[1],idealPos[1]]
    #between p3 and ideal
    l5x[1]=[point3T2[0],idealPos[0]]
    l5y[1]=[point3T2[1],idealPos[1]]
    #between p4 and ideal
    l6x[1]=[point4T2[0],idealPos[0]]
    l6y[1]=[point4T2[1],idealPos[1]]
    #plot all of them
    """ ax.plot(l1x[1],l1y[1],lz)
    ax.plot(l2x[1],l2y[1],lz)
    ax.plot(l3x[1],l3y[1],lz)
    ax.plot(l4x[1],l4y[1],lz)
    ax.plot(l5x[1],l5y[1],lz)
    ax.plot(l6x[1],l6y[1],lz) """

    #trajectory 3

    """ PLOT!!!
    fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z') """

    idealPosXfloor=math.floor(idealPos[0])
    idealPosXceil=math.ceil(idealPos[0])
    idealPosYfloor=math.floor(idealPos[1])
    idealPosYceil=math.ceil(idealPos[1])

    yinIdealXfloor=[]
    yinIdealXceil=[]
    xinIdealYfloor=[]
    xinIdealYceil=[]
    for j in pointsArea:
        if(j[0]==idealPosXfloor):
            yinIdealXfloor.append(j[1])
        if(j[0]==idealPosXceil):
            yinIdealXceil.append(j[1])

    if(len(yinIdealXfloor)>len(yinIdealXceil)):
        xidealPosR=idealPosXfloor
    if(len(yinIdealXfloor)<len(yinIdealXceil)):
        xidealPosR=idealPosXceil
    if(len(yinIdealXceil)==len(yinIdealXfloor)):
        xidealPosR=round(idealPos[0])

    for j in pointsArea:
        if(j[1]==idealPosYfloor):
            xinIdealYfloor.append(j[0])
        if(j[1]==idealPosYceil):
            xinIdealYceil.append(j[0])

    if(len(xinIdealYfloor)>len(xinIdealYceil)):
        yidealPosR=idealPosYfloor
    if(len(xinIdealYfloor)<len(xinIdealYceil)):
        yidealPosR=idealPosYceil
    if(len(xinIdealYceil)==len(xinIdealYfloor)):
        yidealPosR=round(idealPos[1])

    yinXideal=[]
    xinYideal=[]

    for j in pointsArea:
        if(j[0]==xidealPosR):
            yinXideal.append(j[1])
        if(j[1]==yidealPosR):
            xinYideal.append(j[0])

    point1T3=[xidealPosR,max(yinXideal)]
    point2T3=[min(xinYideal),yidealPosR]
    point3T3=[max(xinYideal),yidealPosR]
    point4T3=[xidealPosR,min(yinXideal)]

    idealPosR=[xidealPosR,yidealPosR]

    distanceT3=calculateDistanceT(point1T3,point2T3,point3T3,point4T3,idealPosR)

    #define lines 
    lz=[desiredAltitude,desiredAltitude]
    #between p1 and p2
    l1x[2]=[point1T3[0],point2T3[0]]
    l1y[2]=[point1T3[1],point2T3[1]]
    #between p3 and p4
    l2x[2]=[point3T3[0],point4T3[0]]
    l2y[2]=[point3T3[1],point4T3[1]]
    #between p1 and ideal
    l3x[2]=[point1T3[0],idealPos[0]]
    l3y[2]=[point1T3[1],idealPos[1]]
    #between p2 and ideal
    l4x[2]=[point2T3[0],idealPos[0]]
    l4y[2]=[point2T3[1],idealPos[1]]
    #between p3 and ideal
    l5x[2]=[point3T3[0],idealPos[0]]
    l5y[2]=[point3T3[1],idealPos[1]]
    #between p4 and ideal
    l6x[2]=[point4T3[0],idealPos[0]]
    l6y[2]=[point4T3[1],idealPos[1]]
    #plot all of them
    """ PLOT!!!
    ax.plot(l1x[2],l1y[2],lz)
    ax.plot(l2x[2],l2y[2],lz)
    ax.plot(l3x[2],l3y[2],lz)
    ax.plot(l4x[2],l4y[2],lz)
    ax.plot(l5x[2],l5y[2],lz)
    ax.plot(l6x[2],l6y[2],lz)
    ax.legend() """

    print("Distance T1: ",distanceT1)
    print("Distance T2: ",distanceT2)
    print("Distance T3: ",distanceT3)

    #trajectory plot for the chosen path
    chosen=0
    maxDistance=max(distanceT1,distanceT2,distanceT3)

    if(maxDistance==distanceT1):
        chosen=0
        point1F=point1T1[0:2]
        point2F=point2T1[0:2]
        point3F=point3T1[0:2]
        point4F=point4T1[0:2]
    if(maxDistance==distanceT2):
        chosen=1
        point1F=point1T2[0:2]
        point2F=point2T2[0:2]
        point3F=point3T2[0:2]
        point4F=point4T2[0:2]
        
    if(maxDistance==distanceT3):
        chosen=2
        point1F=point1T3[0:2]
        point2F=point2T3[0:2]
        point3F=point3T3[0:2]
        point4F=point4T3[0:2]

    print("Central Point : ", idealPos)
    print("Point 1 : ", point1F)
    print("Point 2 : ", point2F)
    print("Point 3 : ", point3F)
    print("Point 4 : ", point4F)

    '''fig = plt.figure()

    ax = plt.axes(projection='3d')
    ax.set_title('Trajectory Line Area %d' %(n+1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')'''

    '''ax.plot(l3x[chosen],l3y[chosen],lz,label="Pc->P1")
    ax.plot(l1x[chosen],l1y[chosen],lz,label="P1->P2")
    ax.plot(l4x[chosen],l4y[chosen],lz,label="P2->Pc")
    ax.plot(l5x[chosen],l5y[chosen],lz,label="Pc->P3")
    ax.plot(l2x[chosen],l2y[chosen],lz,label="P3->P4")
    ax.plot(l6x[chosen],l6y[chosen],lz,label="P4->Pc")
    ax.legend()'''

    print("Trajectory ",chosen+1, "was selected")
    #end of PREP
  
    print("Total Distance: ", maxDistance)
    
    print("\n---------------------------- ROTARY-WING ------------------------")
    r_inf=float('inf')
    velocityStraight=optimize.fmin(P_rotary,0,args=(r_inf,))
    powerStraight=P_rotary(velocityStraight[0],r_inf)
    powerHover=P_rotary(0,r_inf)

    print("Minumum Power Velocity=" + str(velocityStraight[0]))
    print("Minimum Power=" + str(powerStraight))
    print("Hover Power=" + str(powerHover))

    timeTrajectory=maxDistance/velocityStraight[0]
    powerConsumed=timeTrajectory*powerStraight+4*powerHover
    ifHovering=(timeTrajectory+4)*powerHover
    
    # Energy consumed per Hour Trajectories
    TEH_Trajectory=TEH_Trajectory+3600*powerConsumed/(timeTrajectory+4)
    
    # Energy consumed per Hour Hovering
    TEH_Hovering=TEH_Hovering+3600*powerHover

    # Energy Consumed per Hour Optimal
    TEH_Optimal_Rotary=TEH_Optimal_Rotary+3600*powerStraight
    
    # Energy Consumed per Hour Calculations
    Best_flag=0
    print("Radius: ", r)
    if r > 0:
        print("\n------ CIRCULAR ------")
        minVelocity=optimize.fmin(P_rotary,0,args=(r,))
        minPower=P_rotary(minVelocity[0],r)

        circularPower=P_rotary(minVelocity, r)
        print("Cirular Min Power Velocity: ", minVelocity[0])
        print("Circular Power: ", circularPower)

        TEH_Circular=TEH_Circular+3600*circularPower
        TEH_SUPPLY_Aux=3600*circularPower
        Best_flag=1

        #Write Circular Trajectory (coords at every second for 1 hour)
        T_Circular=2*np.pi*r/minVelocity[0] #Time to complete a full circle
        
        print("Circular Time: ", T_Circular)
        
        A_velocity = 2 * np.pi / T_Circular # Angular velocity

        time=120
        
        theta = np.linspace(30, time, time-29)*A_velocity # Angle for each timestep 1s

        x = idealPos[0] + r * np.cos(theta)
        y = idealPos[1] + r * np.sin(theta)
        
        #Plot
        '''ax.plot(x, y, desiredAltitude)
        ax.set_box_aspect([25,40,10])
        ax.set_aspect('equal', adjustable='box')'''

        circular.write("FAP: "+str(n+1)+"\n")

        for t in range(30, time):
            circular.write(str(t) +","+ str(round(x[t-30],2)) +","+ str(round(y[t-30],2)) +"\n")        


        print("\n-------- OVAL (CIRCULAR) --------")
        ratio=3/10
        r2=r*ratio
        
        print("r2: ", r2)

        coords=[]
        distances2=[]
        
        for j in perimeter:
            for w in perimeter:
                coords.append((j[0], j[1], w[0], w[1]))
                distances2.append(Euclidean_distance(j[0], j[1], w[0], w[1]))
                
        max_dist=max(distances2)
        index=distances2.index(max_dist)

        if r2 > 0:
            #Slope of the line that connects the most distant points in the perimeter
            slope=math.atan((coords[index][3]-coords[index][1])/(coords[index][2]-coords[index][0]))

            print("ideal: ", idealPos)

            #First Semicircle
            theta3 = np.linspace(slope-np.pi/2, slope+np.pi/2, 101)

            xx1 = np.array(idealPos[0]+(r-r2)*np.cos(slope) + r2 * np.cos(theta3))
            yy1 = np.array(idealPos[1]+(r-r2)*np.sin(slope) + r2 * np.sin(theta3))
            
            #Second Semicircle
            theta2 = np.linspace(slope-np.pi/2, slope+np.pi/2, 101)

            xx2 = np.array(idealPos[0]-(r-r2)*np.cos(slope) - r2 * np.cos(theta2))
            yy2 = np.array(idealPos[1]-(r-r2)*np.sin(slope) - r2 * np.sin(theta2))

            #Straight Lines
            s1x=[xx2[-1], xx1[0]]
            s1y=[yy2[-1], yy1[0]]

            s2x=[xx1[-1], xx2[0]]
            s2y=[yy1[-1], yy2[0]]

            #Plot Points
            '''ax.scatter(xx2[-1], yy2[-1], desiredAltitude, color='r')
            ax.scatter(xx1[0], yy1[0], desiredAltitude, color='g')
            ax.scatter(xx1[25], yy1[25], desiredAltitude, color='g')
            ax.scatter(xx1[50], yy1[50], desiredAltitude, color='g')
            ax.scatter(xx1[75], yy1[75], desiredAltitude, color='g')
            ax.scatter(xx1[-1], yy1[-1], desiredAltitude, color='m')
            ax.scatter(xx2[0], yy2[0], desiredAltitude, color='y')
            ax.scatter(xx2[25], yy2[25], desiredAltitude, color='y')
            ax.scatter(xx2[50], yy2[50], desiredAltitude, color='y')
            ax.scatter(xx2[75], yy2[75], desiredAltitude, color='y')'''

            
            xx=np.concatenate((s1x, xx1, s2x, xx2))
            yy=np.concatenate((s1y, yy1, s2y, yy2))

            #ax.plot(xx, yy, desiredAltitude, color='orange')
            #ax.set_aspect('equal', adjustable='box')
        
            minVelocity=optimize.fmin(P_rotary,0,args=(r2,))
            power=P_rotary(minVelocity, r2)

            print("Curve Min Power Velocity: ", minVelocity[0])
            print("Curve Power: ", power)
        
            distanceCurve=2*np.pi*r2
            distanceStraight=2*(2*r-2*r2)
            print("Curve Distance:", distanceCurve, "Straight Distance:", distanceStraight)

            timeCurve=distanceCurve/minVelocity[0]
            timeStraight=distanceStraight/velocityStraight[0]
            print("Curve Time :", timeCurve, "Straight Time:", timeStraight)

            energyConsumed=timeCurve*power+timeStraight*powerStraight

            timeOval=timeCurve+timeStraight

            powerOval=energyConsumed/timeOval

            print("Oval Power:", powerOval)
            energy=3600*powerOval

            TEH_Oval_Circular=TEH_Oval_Circular+energy

            if energy < TEH_SUPPLY_Aux: 
                TEH_SUPPLY_Aux = energy
                Best_flag=2

            t=30
            oval_c.write("FAP: "+str(n+1)+"\n")

            while t<120:
                oval_c.write(str(round(t,2)) +","+ str(round(xx2[-1],2)) +","+ str(round(yy2[-1],2))+"\n")
                t+=timeStraight/2
                oval_c.write(str(round(t,2)) +","+ str(round(xx1[0],2)) +","+ str(round(yy1[0],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c.write(str(round(t,2)) +","+ str(round(xx1[25],2)) +","+ str(round(yy1[25],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c.write(str(round(t,2)) +","+ str(round(xx1[50],2)) +","+ str(round(yy1[50],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c.write(str(round(t,2)) +","+ str(round(xx1[75],2)) +","+ str(round(yy1[75],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c.write(str(round(t,2)) +","+ str(round(xx1[-1],2)) +","+ str(round(yy1[-1],2)) +"\n")
                t+=timeStraight/2
                oval_c.write(str(round(t,2)) +","+ str(round(xx2[0],2)) +","+ str(round(yy2[0],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c.write(str(round(t,2)) +","+ str(round(xx2[25],2)) +","+ str(round(yy2[25],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c.write(str(round(t,2)) +","+ str(round(xx2[50],2)) +","+ str(round(yy2[50],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c.write(str(round(t,2)) +","+ str(round(xx2[75],2)) +","+ str(round(yy2[75],2)) +"\n")
                t+=(timeCurve/2)/4
            
        else:
            print("Hovering") 
            energy=3600*powerHover
            TEH_Oval_Circular=TEH_Oval_Circular+energy


        print("\n-------- OVAL (AREA) --------")
        idealPos[0]=(coords[index][2]+coords[index][0])/2
        idealPos[1]=(coords[index][3]+coords[index][1])/2

        #Slop of the line that connects the most distant points in the perimeter
        slope=math.atan((coords[index][3]-coords[index][1])/(coords[index][2]-coords[index][0]))

        #Defines curve radius r2 as the smallest distance from the line segment to the points of the perimeter (the points that define the line are not taken into account)

        #ax.scatter(idealPos[0]-((max_dist)/2)*np.cos(slope), idealPos[1]-((max_dist)/2)*np.sin(slope), desiredAltitude)
        #ax.scatter(idealPos[0]+((max_dist)/2)*np.cos(slope), idealPos[1]+((max_dist)/2)*np.sin(slope), desiredAltitude)

        distances3=[]

        for j in perimeter:
            if (j[0],j[1])!=(coords[index][0],coords[index][1]) and (j[0],j[1])!=(coords[index][2],coords[index][3]):
                distances3.append(minDistance([idealPos[0]-((max_dist)/2)*np.cos(slope), idealPos[1]-((max_dist)/2)*np.sin(slope)], [idealPos[0]+((max_dist)/2)*np.cos(slope), idealPos[1]+((max_dist)/2)*np.sin(slope)], [j[0],j[1]]))            

        r2=min(distances3)

        print("r2:", r2)
        if r2 > 0:
            #First Semicircle
            theta3 = np.linspace(slope-np.pi/2, slope+np.pi/2, 101)

            xx1 = np.array(idealPos[0]+(max_dist/2-r2)*np.cos(slope) + r2 * np.cos(theta3))
            yy1 = np.array(idealPos[1]+(max_dist/2-r2)*np.sin(slope) + r2 * np.sin(theta3))

            #Second Semicircle
            theta2 = np.linspace(slope-np.pi/2, slope+np.pi/2, 101)

            xx2 = np.array(idealPos[0]-(max_dist/2-r2)*np.cos(slope) - r2 * np.cos(theta2))
            yy2 = np.array(idealPos[1]-(max_dist/2-r2)*np.sin(slope) - r2 * np.sin(theta2))

            #Straight Lines
            s1x=[xx2[-1], xx1[0]]
            s1y=[yy2[-1], yy1[0]]

            s2x=[xx1[-1], xx2[0]]
            s2y=[yy1[-1], yy2[0]]

            #Plot Points
            '''ax.scatter(xx2[-1], yy2[-1], desiredAltitude, color='r')
            ax.scatter(xx1[0], yy1[0], desiredAltitude, color='g')
            ax.scatter(xx1[25], yy1[25], desiredAltitude, color='g')
            ax.scatter(xx1[50], yy1[50], desiredAltitude, color='g')
            ax.scatter(xx1[75], yy1[75], desiredAltitude, color='g')
            ax.scatter(xx1[-1], yy1[-1], desiredAltitude, color='m')
            ax.scatter(xx2[0], yy2[0], desiredAltitude, color='y')
            ax.scatter(xx2[25], yy2[25], desiredAltitude, color='y')
            ax.scatter(xx2[50], yy2[50], desiredAltitude, color='y')
            ax.scatter(xx2[75], yy2[75], desiredAltitude, color='y')'''

            xx=np.concatenate((s1x, xx1, s2x, xx2))
            yy=np.concatenate((s1y, yy1, s2y, yy2))
            
            #ax.plot(xx, yy, desiredAltitude, color='r')
            #ax.set_aspect('equal', adjustable='box')

            minVelocity=optimize.fmin(P_rotary,0,args=(r2,))
            power=P_rotary(minVelocity, r2)

            print("Curve Min Power Velocity: ", minVelocity[0])
            print("Curve Power: ", power)
        
            distanceCurve=2*np.pi*r2
            distanceStraight=2*(2*(max_dist)/2-2*r2)
            print("Curve Distance:", distanceCurve, "Straight Distance:", distanceStraight)

            timeCurve=distanceCurve/minVelocity[0]
            timeStraight=distanceStraight/velocityStraight[0]
            print("Curve Time :", timeCurve, "Straight Time:", timeStraight)

            energyConsumed=timeCurve*power+timeStraight*powerStraight

            timeOval=timeCurve+timeStraight

            powerOval=energyConsumed/timeOval
            print("Oval Power:", powerOval)

            energy=3600*powerOval
            TEH_Oval_Area=TEH_Oval_Area+energy
            
            if energy < TEH_SUPPLY_Aux: 
                TEH_SUPPLY_Aux = energy
                Best_flag=3

            t=30
            oval_a.write("FAP: "+str(n+1)+"\n")
            while t<=120:
                oval_a.write(str(round(t,2)) +","+ str(round(xx2[-1],2)) +","+ str(round(yy2[-1],2))+"\n")
                t+=timeStraight/2
                oval_a.write(str(round(t,2)) +","+ str(round(xx1[0],2)) +","+ str(round(yy1[0],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a.write(str(round(t,2)) +","+ str(round(xx1[25],2)) +","+ str(round(yy1[25],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a.write(str(round(t,2)) +","+ str(round(xx1[50],2)) +","+ str(round(yy1[50],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a.write(str(round(t,2)) +","+ str(round(xx1[75],2)) +","+ str(round(yy1[75],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a.write(str(round(t,2)) +","+ str(round(xx1[-1],2)) +","+ str(round(yy1[-1],2)) +"\n")
                t+=timeStraight/2
                oval_a.write(str(round(t,2)) +","+ str(round(xx2[0],2)) +","+ str(round(yy2[0],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a.write(str(round(t,2)) +","+ str(round(xx2[25],2)) +","+ str(round(yy2[25],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a.write(str(round(t,2)) +","+ str(round(xx2[50],2)) +","+ str(round(yy2[50],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a.write(str(round(t,2)) +","+ str(round(xx2[75],2)) +","+ str(round(yy2[75],2)) +"\n")
                t+=(timeCurve/2)/4

        else:
            print("Hovering")
            energy=3600*powerHover
            TEH_Oval_Area=TEH_Oval_Area+energy

    else: 
        print("\n------ CIRCULAR ------")
        print("Hovering!!")
        TEH_Circular=TEH_Circular+3600*powerHover
        print("\n-------- OVAL (CIRCULAR) --------")
        print("Hovering!!")
        TEH_Oval_Circular=TEH_Oval_Circular+3600*powerHover
        print("\n-------- OVAL (AREA) -----")
        print("Hovering!!")
        TEH_Oval_Area=TEH_Oval_Area+3600*powerHover

        TEH_SUPPLY_Aux=3600*powerHover
    
    TEH_SUPPLY_ROTARY+=TEH_SUPPLY_Aux

    if Best_flag==0:
        print("Best: None")
    elif Best_flag==1:
        print("Best: Circular")
    elif Best_flag==2:
        print("Best: Oval(circular)")
    elif Best_flag==3:
        print("Best: Oval(area)")

    # -------------------------------------------------------------
        
    print("\n--------------------------- FIXED-WING ------------------------")
    
    r_inf=float('inf')
    velocityStraight=optimize.fmin(P_fixed,0,args=(r_inf,))
    powerStraight=P_fixed(velocityStraight[0],r_inf)
    
    print("Minimum Power Velocity = " + str(velocityStraight[0]))
    print("Minimum Power = " + str(powerStraight))

    # Energy Consumed per Hour Optimal
    TEH_Optimal_Fixed=TEH_Optimal_Fixed+3600*powerStraight
    
    # Energy Consumed per Hour Calculations
    Best_flag=0

    if r >= 5:
        print("\n------ CIRCULAR ------")
        print("Radius: ", r)

        minVelocity=optimize.fmin(P_fixed,0,args=(r,))
        minPower=P_fixed(minVelocity[0],r)

        circularPower=P_fixed(minVelocity, r)[0]
        print("Cirular Min Power Velocity: ", minVelocity[0])
        print("Circular Power: ", circularPower)

        TEH_Circular=TEH_Circular+3600*circularPower
        TEH_SUPPLY_Aux=3600*circularPower
        Best_flag=1

        #Write Circular Trajectory (coords at every second for 1 hour)
        T_Circular=2*np.pi*r/minVelocity[0] #Time to complete a full circle
        
        print("Circular Time: ", T_Circular)
        
        A_velocity = 2 * np.pi / T_Circular # Angular velocity

        time=120
        
        theta = np.linspace(30, time, time-29)*A_velocity # Angle for each timestep 1s

        x = idealPos[0] + r * np.cos(theta)
        y = idealPos[1] + r * np.sin(theta)

        #Plot
        '''ax.plot(x, y, desiredAltitude)
        ax.set_box_aspect([25,40,10])
        ax.set_aspect('equal', adjustable='box')'''
        
        circular_fixed.write("FAP: "+str(n+1)+"\n")
        for t in range(30, time):
            circular_fixed.write(str(t) +","+ str(round(x[t-30],2)) +","+ str(round(y[t-30],2)) +"\n")        


        print("\n-------- OVAL (CIRCULAR) --------")
        ratio=3/10
        r2=r*ratio
        
        print("r2: ", r2)

        coords=[]
        distances2=[]
            
        for j in perimeter:
            for w in perimeter:
                coords.append((j[0], j[1], w[0], w[1]))
                distances2.append(Euclidean_distance(j[0], j[1], w[0], w[1]))
        
        max_dist=max(distances2)
        index=distances2.index(max_dist)  

        if r2 >= 5:
            #Slope of the line that connects the most distant points in the perimeter
            slope=math.atan((coords[index][3]-coords[index][1])/(coords[index][2]-coords[index][0]))

            print("ideal: ", idealPos)

            #First Semicircle
            theta3 = np.linspace(slope-np.pi/2, slope+np.pi/2, 101)

            xx1 = np.array(idealPos[0]+(r-r2)*np.cos(slope) + r2 * np.cos(theta3))
            yy1 = np.array(idealPos[1]+(r-r2)*np.sin(slope) + r2 * np.sin(theta3))
            
            #Second Semicircle
            theta2 = np.linspace(slope-np.pi/2, slope+np.pi/2, 101)

            xx2 = np.array(idealPos[0]-(r-r2)*np.cos(slope) - r2 * np.cos(theta2))
            yy2 = np.array(idealPos[1]-(r-r2)*np.sin(slope) - r2 * np.sin(theta2))

            #Straight Lines
            s1x=[xx2[-1], xx1[0]]
            s1y=[yy2[-1], yy1[0]]

            s2x=[xx1[-1], xx2[0]]
            s2y=[yy1[-1], yy2[0]]
            
            xx=np.concatenate((s1x, xx1, s2x, xx2))
            yy=np.concatenate((s1y, yy1, s2y, yy2))

            minVelocity=optimize.fmin(P_fixed,0,args=(r2,))
            power=P_fixed(minVelocity, r2)[0]

            print("Curve Min Power Velocity: ", minVelocity[0])
            print("Curve Power: ", power)
        
            distanceCurve=2*np.pi*r2
            distanceStraight=2*(2*r-2*r2)
            print("Curve Distance:", distanceCurve, "Straight Distance:", distanceStraight)

            timeCurve=distanceCurve/minVelocity[0]
            timeStraight=distanceStraight/velocityStraight[0]
            print("Curve Time :", timeCurve, "Straight Time:", timeStraight)

            energyConsumed=timeCurve*power+timeStraight*powerStraight

            timeOval=timeCurve+timeStraight

            powerOval=energyConsumed/timeOval

            print("Oval Power:", powerOval)
            energy=3600*powerOval

            TEH_Oval_Circular=TEH_Oval_Circular+energy

            if energy < TEH_SUPPLY_Aux: 
                TEH_SUPPLY_Aux = energy
                Best_flag=2

            t=30
            oval_c_fixed.write("FAP: "+str(n+1)+"\n")

            while t<=120:
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx2[-1],2)) +","+ str(round(yy2[-1],2))+"\n")
                t+=timeStraight/2
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx1[0],2)) +","+ str(round(yy1[0],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx1[25],2)) +","+ str(round(yy1[25],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx1[50],2)) +","+ str(round(yy1[50],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx1[75],2)) +","+ str(round(yy1[75],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx1[-1],2)) +","+ str(round(yy1[-1],2)) +"\n")
                t+=timeStraight/2
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx2[0],2)) +","+ str(round(yy2[0],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx2[25],2)) +","+ str(round(yy2[25],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx2[50],2)) +","+ str(round(yy2[50],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_c_fixed.write(str(round(t,2)) +","+ str(round(xx2[75],2)) +","+ str(round(yy2[75],2)) +"\n")
                t+=(timeCurve/2)/4

        else:
            print("OVAL (CIRCULAR) impossible for fixed-wing UAV Group %d" %(n+1))

        print("\n-------- OVAL (AREA) --------")
        idealPos[0]=(coords[index][2]+coords[index][0])/2
        idealPos[1]=(coords[index][3]+coords[index][1])/2

        #Slop of the line that connects the most distant points in the perimeter
        slope=math.atan((coords[index][3]-coords[index][1])/(coords[index][2]-coords[index][0]))

        #Defines curve radius r2 as the smallest distance from the line segment to the points of the perimeter (the points that define the line are not taken into account)

        #ax.scatter(idealPos[0]-((max_dist)/2)*np.cos(slope), idealPos[1]-((max_dist)/2)*np.sin(slope), desiredAltitude)
        #ax.scatter(idealPos[0]+((max_dist)/2)*np.cos(slope), idealPos[1]+((max_dist)/2)*np.sin(slope), desiredAltitude)

        distances3=[]

        for j in perimeter:
            if (j[0],j[1])!=(coords[index][0],coords[index][1]) and (j[0],j[1])!=(coords[index][2],coords[index][3]):
                distances3.append(minDistance([idealPos[0]-((max_dist)/2)*np.cos(slope), idealPos[1]-((max_dist)/2)*np.sin(slope)], [idealPos[0]+((max_dist)/2)*np.cos(slope), idealPos[1]+((max_dist)/2)*np.sin(slope)], [j[0],j[1]]))            

        r2=min(distances3)

        print("r2:", r2)
        if r2 >= 5:
            #First Semicircle
            theta3 = np.linspace(slope-np.pi/2, slope+np.pi/2, 101)

            xx1 = np.array(idealPos[0]+(max_dist/2-r2)*np.cos(slope) + r2 * np.cos(theta3))
            yy1 = np.array(idealPos[1]+(max_dist/2-r2)*np.sin(slope) + r2 * np.sin(theta3))

            #Second Semicircle
            theta2 = np.linspace(slope-np.pi/2, slope+np.pi/2, 101)

            xx2 = np.array(idealPos[0]-(max_dist/2-r2)*np.cos(slope) - r2 * np.cos(theta2))
            yy2 = np.array(idealPos[1]-(max_dist/2-r2)*np.sin(slope) - r2 * np.sin(theta2))

            #Straight Lines
            s1x=[xx2[-1], xx1[0]]
            s1y=[yy2[-1], yy1[0]]

            s2x=[xx1[-1], xx2[0]]
            s2y=[yy1[-1], yy2[0]]

            xx=np.concatenate((s1x, xx1, s2x, xx2))
            yy=np.concatenate((s1y, yy1, s2y, yy2))
            
            minVelocity=optimize.fmin(P_fixed,0,args=(r2,))
            power=P_fixed(minVelocity, r2)[0]

            print("Curve Min Power Velocity: ", minVelocity[0])
            print("Curve Power: ", power)
        
            distanceCurve=2*np.pi*r2
            distanceStraight=2*(2*(max_dist)/2-2*r2)
            print("Curve Distance:", distanceCurve, "Straight Distance:", distanceStraight)

            timeCurve=distanceCurve/minVelocity[0]
            timeStraight=distanceStraight/velocityStraight[0]
            print("Curve Time :", timeCurve, "Straight Time:", timeStraight)

            energyConsumed=timeCurve*power+timeStraight*powerStraight

            timeOval=timeCurve+timeStraight

            powerOval=energyConsumed/timeOval
            print("Oval Power:", powerOval)

            energy=3600*powerOval
            TEH_Oval_Area=TEH_Oval_Area+energy
            
            if energy < TEH_SUPPLY_Aux: 
                TEH_SUPPLY_Aux = energy
                Best_flag=3

            t=30
            oval_a.write("FAP: "+str(n+1)+"\n")

            while t<=120:
                oval_a_fixed.write(str(round(t,2)) +","+ str(round(xx2[-1],2)) +","+ str(round(yy2[-1],2))+"\n")
                t+=timeStraight/2
                oval_a_fixed.write(str(round(t,2)) +","+ str(round(xx1[0],2)) +","+ str(round(yy1[0],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a_fixed.write(str(round(t,2)) +","+ str(round(xx1[25],2)) +","+ str(round(yy1[25],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a_fixed.write(str(round(t,2)) +","+ str(round(xx1[50],2)) +","+ str(round(yy1[50],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a_fixed.write(str(round(t,2)) +","+ str(round(xx1[75],2)) +","+ str(round(yy1[75],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a_fixed.write(str(round(t,2)) +","+ str(round(xx1[-1],2)) +","+ str(round(yy1[-1],2)) +"\n")
                t+=timeStraight/2
                oval_a_fixed.write(str(round(t,2)) +","+ str(round(xx2[0],2)) +","+ str(round(yy2[0],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a.write(str(round(t,2)) +","+ str(round(xx2[25],2)) +","+ str(round(yy2[25],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a_fixed.write(str(round(t,2)) +","+ str(round(xx2[50],2)) +","+ str(round(yy2[50],2)) +"\n")
                t+=(timeCurve/2)/4
                oval_a_fixed.write(str(round(t,2)) +","+ str(round(xx2[75],2)) +","+ str(round(yy2[75],2)) +"\n")
                t+=(timeCurve/2)/4
        else:
            print("OVAL (AREA) impossible for fixed-wing UAV Group %d" %(n+1))

    else: 
        print("Imposible trajectories for fixed-wing UAV Group %d" %(n+1))
        TEH_SUPPLY_Aux=float('inf')

    TEH_SUPPLY_FIXED+=TEH_SUPPLY_Aux

    if Best_flag==0:
        print("Best: None")
    elif Best_flag==1:
        print("Best: Circular")
    elif Best_flag==2:
        print("Best: Oval(circular)")
    elif Best_flag==3:
        print("Best: Oval(area)")

# PLOT ALL AREAS

'''fig = plt.figure()

ax = plt.axes(projection='3d')
#ax.set_title('All Intersection Areas')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim3d(0, 100)
ax.set_ylim3d(0, 100)
ax.set_zlim3d(0, 10)
z_ticks = np.arange(0, 12, 6)  
ax.set_zticks(z_ticks)


for index,j in enumerate(TotalPoints):
    for n in range(int(nGroups)):
        if n==0:    
            if 0 < index < n_points[n]-1:
                ax.scatter(j[0],j[1],j[2],marker='o', color=colors[n])
        else:
            if (n_points[n-1])-1 < index < n_points[n]-1:
                ax.scatter(j[0],j[1],j[2],marker='o', color=colors[n])


ax.set_box_aspect([100,100,12])

ax.set_aspect('equal', adjustable='box')'''

'''print("Optimal: ", round(TEH_Optimal/1000, 2))
print("Circular: ", round(TEH_Circular/1000, 2))
print("Oval(circular): ", round(TEH_Oval_Circular/1000, 2))
print("Oval(area): ", round(TEH_Oval_Area/1000, 2))
print("SUPPLY: ", round(TEH_SUPPLY/1000, 2))
print("EREP: ", round(TEH_Trajectory/1000, 2))
print("Hovering: ", round(TEH_Hovering/1000, 2))'''

#print("Energy Reduction (%):", round((TEH_Hovering/1000-TEH_SUPPLY/1000)/(TEH_Hovering/1000)*100,1))


#fig = plt.figure(figsize=(8, 4.8))
'''xpart=['Optimal', 'EREP', 'Circular', 'Oval Circular', 'Oval Area', 'SUPPLY', 'Hovering']
ypart=[TEH_Optimal/1000, TEH_Trajectory/1000, TEH_Circular/1000, TEH_Oval_Circular/1000, TEH_Oval_Area/1000, TEH_SUPPLY/1000, TEH_Hovering/1000]
plt.bar(xpart,ypart,color=['black','C4', 'C0', 'orange', 'red', 'green'])
plt.title('Energy Consumption Comparison')
plt.ylabel('Energy Consumed per Hour (KiloJoule)')
'''

print("SUPPLY Rotary: ", round(TEH_SUPPLY_ROTARY/1000, 2))

print("SUPPLY Fixed: ", round(TEH_SUPPLY_FIXED/1000, 2))

print("Energy consumption increase (%):", round((TEH_SUPPLY_FIXED/TEH_SUPPLY_ROTARY-1)*100, 0))

fig = plt.figure(figsize=(9, 5.2))
plt.rcParams['font.size'] = 12
#xpart=['Optimal_Rotary', 'Optimal_Fixed','SUPPLY_Rotary', 'SUPPLY_Fixed']
#ypart=[TEH_Optimal_Rotary/1000, TEH_Optimal_Fixed/1000, TEH_SUPPLY_ROTARY/1000, TEH_SUPPLY_FIXED/1000]
xpart=['SUPPLY Rotary', 'SUPPLY Fixed']
ypart=[TEH_SUPPLY_ROTARY/1000, TEH_SUPPLY_FIXED/1000]
color=['green','C0']
plt.bar(xpart,ypart,color=['green','C0'])
for i in range(len(xpart)):
    plt.text(i, round(ypart[i], 2), str(round(ypart[i], 2)), ha='center', va='bottom', color=color[i])
#plt.title('Energy Consumption Comparison')
plt.ylabel('Energy Consumed per Hour (KiloJoule)',)

'''xpart=['Circular', 'Oval Circular', 'Oval Area', 'SUPPLY', 'Hovering']
ypart=[TEH_Circular/1000, TEH_Oval_Circular/1000, TEH_Oval_Area/1000, TEH_SUPPLY/1000, TEH_Hovering/1000]
plt.title('Energy Consumption Comparison')
plt.ylabel('Energy Consumed per Hour (KiloJoule)')
plt.ylim(bottom=(TEH_Optimal/1000)*0.8, top=(TEH_Hovering/1000)*1.05)

plt.bar(xpart,ypart,color=['C0', 'orange', 'red', 'green', 'black'])'''


'''results = open('Results10.txt', 'a')

results.write(str(nGroups)+","+str(round((TEH_SUPPLY_ROTARY/1000),2))+","+str(round((TEH_SUPPLY_FIXED/1000),2))+"\n")'''

'''sensitivity = open('Alt_Sensitivity2.txt', 'a')
sensitivity.write(str(alt_min)+","+str(round((TEH_Optimal/1000),2))+","+str(round((TEH_Oval_Circular/1000),2))+","+str(round((TEH_Oval_Area/1000),2))+","+str(round((TEH_Oval_Best/1000),2))+","+str(round((TEH_Circular/1000),2))+","+str(round((TEH_Trajectory/1000),2))+","+str(round((TEH_Hovering/1000),2))+"\n")'''

#sensitivity = open('New_Sensitivity2.txt', 'a')
#sensitivity.write(str(alt_min)+","+str(round((TEH_Optimal/1000),2))+","+str(round((TEH_Oval_min/1000),2))+","+str(round((TEH_Oval_const/1000),2))+","+str(round((TEH_Oval_inc/1000),2))+","+str(round((TEH_Circular/1000),2))+","+str(round((TEH_Trajectory/1000),2))+","+str(round((TEH_Hovering/1000),2))+"\n")

#plt.close('all') 
if(flag):
    print("COLISION DETECTED!!!!")
else: print("")

plt.show()

""" 
#SHIFT+ALT+A
"""
