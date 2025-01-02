from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import math
import numpy as np
import matplotlib.pyplot as plt
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
alt_max=6
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

best_flag = {
    0: "Best: None",
    1: "Best: Circular",
    2: "Best: Inner Elliptic",
    3: "Best: Elliptic"
}

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

# Function to return the minimum distance between a line segment AB and a point E
def minDistance(A, B, C) :
 
    # vector AB
    AB = [None, None]
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]
 
    # vector BP
    BC = [None, None]
    BC[0] = C[0] - B[0]
    BC[1] = C[1] - B[1]
 
    # vector AP
    AC = [None, None]
    AC[0] = C[0] - A[0]
    AC[1] = C[1] - A[1]
 
    # Variables to store dot product
 
    # Calculating the dot product
    AB_BC = AB[0] * BC[0] + AB[1] * BC[1]
    AB_AC = AB[0] * AC[0] + AB[1] * AC[1]
 
    # Minimum distance from point C to the line segment
    reqAns = 0
 
    # Case 1
    if (AB_BC > 0) :
 
        # Finding the magnitude
        y = C[1] - B[1]
        x = C[0] - B[0]
        reqAns = math.sqrt(x * x + y * y)
 
    # Case 2
    elif (AB_AC < 0) :
        y = C[1] - A[1]
        x = C[0] - A[0]
        reqAns = math.sqrt(x * x + y * y)
 
    # Case 3
    else:
        # Finding the perpendicular distance
        x1 = AB[0]
        y1 = AB[1]
        x2 = AC[0]
        y2 = AC[1]
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

#Dictionary for SNR and Data Rate relation for WiFi 802.11ac
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

def processGUData(f, nGUs_group, n, nGUs):
    GUs = []
    x = []
    y = []
    z = []
    traffic = []

    f.readline()  # Skip the header line
    for j in range(nGUs_group[n]):
        GUs.append(f.readline().split(","))
        x.append(float(GUs[j][0]))
        y.append(float(GUs[j][1]))
        z.append(float(GUs[j][2]))
        traffic.append(float(GUs[j][3]))

        if traffic[j] > (maxMCS / nGUs):
            traffic[j] = maxMCS / nGUs

    return GUs, x, y, z, traffic

def map_traffic_to_snr(traffic, dicMCS, SNR_MARGIN):
    data_rate_val = []

    for t in traffic:
        # Find the minimum valid data rate that satisfies the condition
        valid_data_rates = [dicMCS[i].get("data_rate") for i in range(10) if dicMCS[i].get("data_rate") >= t]
        data_rate_val.append(min(valid_data_rates) if valid_data_rates else None)

    SNR_values = []

    for rate in data_rate_val:
        # Find the SNR value corresponding to the data rate
        if rate is not None:
            for entry in dicMCS:
                if entry.get("data_rate") == rate:
                    SNR_values.append(entry.get("SNR") + SNR_MARGIN)
                    break

    return SNR_values

def calculate_valid_points(x, y, z, xToCalcPos, yToCalcPos, xToCalcNeg, yToCalcNeg, alt_max, alt_min, step, Pt, noise, SNR_values, c, freq):
    """
    Calculate valid points where all SNR values are greater than or equal to the threshold.

    Parameters:
    x, y, z : list
        Coordinates of points.
    xToCalcPos, yToCalcPos : list
        Positive range for x and y.
    xToCalcNeg, yToCalcNeg : list
        Negative range for x and y.
    alt_max : float
        Maximum altitude.
    alt_min : float
        Minimum altitude.
    step : float
        Increment step for grid calculation.
    Pt : float
        Transmitted power.
    noise : float
        Noise.
    SNR_values : list
        Threshold SNR values for each point.
    c : float
        Speed of light.
    freq : float
        Frequency of the signal.

    Returns:
    valid_points : list
        List of valid points meeting the SNR conditions.
    """
    # Precompute coordinates of known points
    pd = [np.array((x[i], y[i], z[i])) for i in range(len(x))]

    # Define bounds for the grid
    xmax, ymax, zmax = max(xToCalcPos), math.floor(max(yToCalcPos)), alt_max
    xmin, ymin, zmin = min(xToCalcNeg), min(yToCalcNeg), alt_min

    valid_points = []

    # Iterate through the grid
    xd = xmin
    while xd <= xmax:
        yd = ymin
        while yd <= ymax:
            zd = zmin
            while zd <= zmax:
                current_point = np.array((xd, yd, zd))
                
                # Calculate SNR for all points
                count = 0
                for i in range(len(pd)):
                    dist = np.linalg.norm(pd[i] - current_point)
                    
                    if dist == 0:
                        Pr = Pt
                    else:
                        Pr = Pt + 20 * math.log10(c / (4 * freq * dist * math.pi))

                    if (Pr - noise) >= SNR_values[i]:
                        count += 1

                # If all points meet the condition, add the current point
                if count == len(pd):
                    valid_points.append(current_point)

                zd += step
            yd += step
        xd += step

    return valid_points

for n in range(int(nGroups)):
    GUs, x, y, z, traffic = processGUData(f, nGUs_group, n, nGUs)
    
    print(x)
    print(y)
    print(z)
    print(traffic)
    
    SNR_values = map_traffic_to_snr(traffic, dicMCS, SNR_MARGIN)
    
    xToCalcPos, xToCalcNeg = [], []
    yToCalcPos, yToCalcNeg = [], []
    zToCalcPos, zToCalcNeg = [], []

    # Iterate through SNR values and calculate positions
    for j, snr in enumerate(SNR_values):
        dist = math.floor(distanceForSNR(snr))

        xToCalcPos.append(x[j] + dist)
        xToCalcNeg.append(x[j] - dist)
        yToCalcPos.append(y[j] + dist)
        yToCalcNeg.append(y[j] - dist)
        zToCalcPos.append(z[j] + dist)
        zToCalcNeg.append(z[j] - dist)

    # Print minimum and maximum positions
    print("\nMin/Max Pos:")
    print(xToCalcPos)
    print(xToCalcNeg)
    print(yToCalcPos)
    print(yToCalcNeg)
    print(zToCalcPos)
    print(zToCalcNeg)
    print("\n")

    '''#Figure for positions

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
    validPoints = calculate_valid_points(x, y, z, xToCalcPos, yToCalcPos, xToCalcNeg, yToCalcNeg, alt_max, alt_min, step, Pt, noise, SNR_values, c, freq)
    
    if(len(validPoints)==0):
        print("No intersection was found")

    print("Number of points: ", len(validPoints))
    print("SNR values = ", SNR_values)

    validAltitudes = [j[2] for j in validPoints]

    # Find the most common altitude
    desiredAltitude = Counter(validAltitudes).most_common(1)[0][0]

    print("Desired Altitude= " + str(desiredAltitude))

    #Plot area for desired altitude (intersection area)
    '''fig = plt.figure()

    ax = plt.axes(projection='3d')
    #ax.set_title('Intersection Area Group %d' %(n+1))
    ax.set_title('Intersection Area', fontsize=22)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    '''
    
    #Points in the desired altitude
      
    pointsArea = [point for point in validPoints if point[2] == desiredAltitude]

    # Uncomment the following line to visualize points if needed
    #for point in pointsArea:
        #ax.scatter(point[0], point[1], point[2], marker='o', color=colors[n])

    #ax.set_xlim(0,100)
    #ax.set_ylim(0,100)
    
    # Verify for area collision -> removes points from new area if they are already in another

    '''ax.set_box_aspect([50,50,15])
    ax.set_aspect('equal', adjustable='box')'''

    j=0
    if TotalPoints: 
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

    print("Ideal Position= " + str(idealPos))

    '''fig = plt.figure()

    ax = plt.axes(projection='3d')
    #ax.set_title('Perimeter and Circular Trajectory Group %d' %(n+1))
    ax.set_title('Circular', fontsize = 22)
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
    
    print("\n---------------------------- ROTARY-WING ------------------------")
    r_inf=float('inf')
    velocityStraight=optimize.fmin(P_rotary, 0, args=(r_inf,))
    powerStraight=P_rotary(velocityStraight[0],r_inf)
    powerHover=P_rotary(0,r_inf)

    print("Minumum Power Velocity=" + str(velocityStraight[0]))
    print("Minimum Power=" + str(powerStraight))
    print("Hover Power=" + str(powerHover))

    # Energy consumed per Hour Hovering
    TEH_Hovering=TEH_Hovering+3600*powerHover

    # Energy Consumed per Hour Optimal
    TEH_Optimal_Rotary=TEH_Optimal_Rotary+3600*powerStraight
    
    # Energy Consumed per Hour Calculations
    Best_flag=0
    print("Radius: ", r)
    if r > 0:
        print("\n------ CIRCULAR ------")
        minVelocity=optimize.fmin(P_rotary, 0, args=(r,))
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


        print("\n-------- INNER-ELLIPTIC --------")
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
        
            minVelocity=optimize.fmin(P_rotary, 0, args=(r2,))
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


        print("\n-------- ELLIPTIC --------")
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
        print("\n-------- INNER-ELLIPTIC --------")
        print("Hovering!!")
        TEH_Oval_Circular=TEH_Oval_Circular+3600*powerHover
        print("\n-------- ELLIPTIC -----")
        print("Hovering!!")
        TEH_Oval_Area=TEH_Oval_Area+3600*powerHover

        TEH_SUPPLY_Aux=3600*powerHover
    
    TEH_SUPPLY_ROTARY+=TEH_SUPPLY_Aux

    print(best_flag.get(Best_flag))

    # -------------------------------------------------------------
        
    print("\n--------------------------- FIXED-WING ------------------------")
    
    r_inf=float('inf')
    velocityStraight=optimize.fmin(P_fixed, 0, args=(r_inf,))
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

        minVelocity=optimize.fmin(P_fixed, 0, args=(r,))
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


        print("\n-------- INNER-ELLIPTIC --------")
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

            minVelocity=optimize.fmin(P_fixed, 0, args=(r2,))
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
            print("INNER-ELLIPTIC impossible for fixed-wing UAV Group %d" %(n+1))

        print("\n-------- ELLIPTIC --------")
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
            print("ELLIPTIC impossible for fixed-wing UAV Group %d" %(n+1))

    else: 
        print("Imposible trajectories for fixed-wing UAV Group %d" %(n+1))
        TEH_SUPPLY_Aux=float('inf')

    TEH_SUPPLY_FIXED+=TEH_SUPPLY_Aux

    # Print best
    print(best_flag.get(Best_flag))

print("\n--------------------------- RESULTS ------------------------")

print("SUPPLY Rotary: ", round(TEH_SUPPLY_ROTARY/1000, 2))

print("SUPPLY Fixed: ", round(TEH_SUPPLY_FIXED/1000, 2))

print("Energy consumption increase (%):", round((TEH_SUPPLY_FIXED/TEH_SUPPLY_ROTARY-1)*100, 0))

fig = plt.figure(figsize=(9, 5.2))
plt.rcParams['font.size'] = 12
xpart=['SUPPLY Rotary', 'SUPPLY Fixed']
ypart=[TEH_SUPPLY_ROTARY/1000, TEH_SUPPLY_FIXED/1000]
color=['green','C0']
plt.bar(xpart,ypart,color=['green','C0'])
for i in range(len(xpart)):
    plt.text(i, round(ypart[i], 2), str(round(ypart[i], 2)), ha='center', va='bottom', color=color[i])

plt.ylabel('Energy Consumed per Hour (KiloJoule)',)

plt.show()
