# Composite Laminate Theory Program
# Paul Blackhurst - BYU - ME EN 456
# April 21, 2023
#
# When running this program, the user can input a desired matrix, failure criterion,
# loading scenario, and number of layers. This is data for the entire beam.
#
# The user can also specify the fiber material, volume fraction, fiber direction, and
# thickness of every single layer. The inputs can be different for each layer if desired.
#
# All outputs are printed to the terminal.
#
# This code reads in data from a fiber_properties.xlsx file and a matrix_properties.xlsx file.
# It also makes use of a layers.csv file where it displays the input info of each ply.
#
# This code requires the following modules imported below.

# Import modules
import tkinter as tk
import pandas as pd
import numpy as np
import csv

#  Create window, set title and size
window = tk.Tk()
window.title("CLT Program")
window.geometry('500x600')

# Entries for entire beam
title1 = tk.Label(window, text='Entries for entire beam', fg='gray')
title1.pack()

line1 = tk.Frame(window, height=1, width=200, bg="red")
line1.pack()

# Matrix drop down menu
matrix = tk.Label(window, text='Select Matrix Material')
matrix.pack()
mat_options = ["LM","IMLS","IMHS","HM","Polyimide","PMR"]
mat_clicked = tk.StringVar()
mat_drop = tk.OptionMenu(window, mat_clicked, *mat_options)
mat_drop.pack()

# Failure Criterion drop down menu
failcrit = tk.Label(window, text='Select Failure Criterion')
failcrit.pack()
failcrit_options = ["Maximum Stress Criterion", "Maximum Strain Criterion", "Tsai-Hill"]
failcrit_clicked = tk.StringVar()
drop2 = tk.OptionMenu(window, failcrit_clicked, *failcrit_options)
drop2.pack()

# Loading case input widget
load = tk.Label(window, text='Input loading scenario as comma delineated list (i.e. (Nx,Ny,Nz,Mx,My,Mz))\n***Enter values as Pa-m***')
load.pack()
bx_load = tk.Entry(window, width=15, borderwidth=3)
bx_load.pack()

# Number of layers input widget
layers = tk.Label(window, text='Number of layers')
layers.pack()
bx_lyrs = tk.Entry(window, width=5, borderwidth=3)
bx_lyrs.pack()

# Entries for entire beam
title2 = tk.Label(window, text='Entries for each layer', fg='gray')
title2.pack()

line2 = tk.Frame(window, height=1, width=200, bg="red")
line2.pack()

#  Fiber drop down menu
fiber = tk.Label(window, text='Select Fiber Material')
fiber.pack()
fib_options = ["Boron","HMS","AS","T300","Kevlar","S-Glass","E-Glass"]
fib_clicked = tk.StringVar()
fib_drop = tk.OptionMenu(window, fib_clicked, *fib_options)
fib_drop.pack()

# Fiber volume fraction input widget
volfrac = tk.Label(window, text='Input Fiber Volume Fraction (i.e. 0.50)')
volfrac.pack()
bx_vf = tk.Entry(window, width=5, borderwidth=3)
bx_vf.pack()

# Layer orientation input widget
orient = tk.Label(window, text="Layer Orientation (degrees)")
orient.pack()
bx_orient = tk.Entry(window, width=5, borderwidth=3)
bx_orient.pack()

# Layer thickness input widget
thick = tk.Label(window, text='Thickness (m)')
thick.pack()
bx_thick = tk.Entry(window, width=5, borderwidth=3)
bx_thick.pack()

main_list=[] # List of values that are appended to layers.csv

def Add(): # Function for add button
   list=[mat_clicked.get(), bx_vf.get(), fib_clicked.get(), bx_orient.get(), bx_thick.get()]
   main_list.append(list)

def Save(): # Function for save button
    with open('layers.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["matrix","volume fraction","fiber","orientation","thickness"])
        writer.writerows(main_list)
    save.configure(text="Saved")

def Clear(): # Function for clear button
   bx_orient.delete(0,tk.END)
   bx_thick.delete(0,tk.END)

# Defining and locating buttons in GUI 
add = tk.Button(window, text="Add", command=Add)
add.pack()
clear = tk.Button(window, text="Clear", command=Clear)
clear.pack()
save = tk.Button(window, text="Save", command=Save)
save.pack()

line3 = tk.Frame(window, height=1, width=200, bg="red")
line3.pack()

# This function is essentially the heart of this CLT program. Here is where
# all calculations are made once the "Calculate" button is pressed.
def Calculate():
    # Predefining some empty lists and arrays that are used later. N is number of layers.
    N = int(bx_lyrs.get())
    Qbar = []
    E2l, E1l , v12l, G12l ,Gml = [], [], [], [], []

    # This for-loop pulls in data and calculates the Qbar matrix for every layer of the beam.
    for i in range(N):
        # Pull in data from the layers.csv file that has layer specific data.
        layer_data = pd.read_csv('/Users/paulblackhurst/Desktop/Python/layers.csv')
        volfrac = layer_data["volume fraction"]
        fiber = layer_data["fiber"]
        orientation = layer_data["orientation"]
        thickness = layer_data["thickness"]

        # Changing data type to a list so I can index from it later.
        vf = volfrac.tolist()
        fib = fiber.tolist()
        theta = orientation.tolist()
        t = thickness.tolist()

        # Getting an index number for getting material properties from the excel files 
        fp = fib_options.index(fib[i])
        mp = mat_options.index(mat_clicked.get())

        fiber_prop = pd.read_excel('/Users/paulblackhurst/Desktop/Python/Homework Assignments/ME EN 456/fiber_properties.xlsx')
        matrix_prop = pd.read_excel('/Users/paulblackhurst/Desktop/Python/Homework Assignments/ME EN 456/matrix_properties.xlsx')
        E1F = fiber_prop.get("Longitudinal modulus") * 10**9
        E2F = fiber_prop.get("Transverse modulus") * 10**9
        v12F = fiber_prop.get("Longitudinal Poisson")
        G12F = fiber_prop.get("Longitudinal shear") * 10**9
        EM = matrix_prop.get("Modulus") * 10**9
        vM = matrix_prop.get("Poisson")
        a = E1F[fp]
        
        # Calculations
        Gm = (EM[mp] / (2 * (1 + vM[mp])))
        E1 = (E1F[fp] * vf[i]) + (EM[mp] * (1 - vf[i])) # Equation 3.27
        E2 = EM[mp] * ((1 - np.sqrt(vf[i])) + (np.sqrt(vf[i]))/(1 - (np.sqrt(vf[i]) * (1 - (EM[mp]/E2F[fp]))))) # Equation 3.54
        v12 = (v12F[fp] * vf[i]) + (vM[mp] * (1 - vf[i]))
        v21 = v12 * (E2/E1)
        G12 = Gm * ((1 - np.sqrt(vf[i])) + (np.sqrt(vf[i])/(1 - (np.sqrt(vf[i]) * (1 - (Gm/G12F[fp]))))))

        E2l.append(E2) # This saves E2 in an array to be used for calculating Stp down below
        E1l.append(E1) # This saves E1 in an array to be used for calculating Slm down below
        v12l.append(v12) # "
        G12l.append(G12) # "
        Gml.append(Gm) # "

        S = np.array([
            [(1/E1), (-v12/E1), 0],
            [(-v21/E2), (1/E2), 0],
            [0, 0, (1/G12)]])
        Q = np.linalg.inv(S)
        
        # T matrices for rotating into the ply frame.
        c = np.cos(theta[i] * (np.pi/180))
        s = np.sin(theta[i] * (np.pi/180))
        T = np.array([[c**2,s**2,2*c*s],[s**2,c**2,-2*c*s],[-c*s,c*s,((c**2)-(s**2))]])
        T2 = np.array([[c**2,s**2,c*s],[s**2,c**2,-c*s],[-2*c*s,2*c*s,((c**2)-(s**2))]])

        Qb = np.linalg.inv(T) @ Q @ T2
        Qbar.append(Qb)

    Qbar = np.array(Qbar)

    ## ABBD Matrices

    # Create an array of z-values defined from the center of the beam (i.e. [-2, -1, 0, 2, 3])
    t_total = np.sum(t)
    z = np.zeros(N+1)
    z[0] = -t_total/2

    for k in range(1, N+1):
        z[k] = z[k-1] + t[k-1]

    # Empty abd matrices
    a = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    b = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    d = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # Calculations
    for i in range(3): # row
        for j in range(3): # column
            for k in range(N): # matrix
                a[i][j] += Qbar[k][i][j] * (z[k+1] - z[k])
                b[i][j] += Qbar[k][i][j] * ((z[k+1]**2)-(z[k]**2))
                d[i][j] += Qbar[k][i][j] * ((z[k+1]**3)-(z[k]**3))

    A = np.array(a)
    B = np.array(b) * 0.5
    D = np.array(d) * (1/3)

    # Combining into a single 6x6 ABBD matrix
    top_row = np.block([[A, B]])
    bottom_row = np.block([[B, D]])
    ABD = np.vstack([top_row, bottom_row])

    ## Stresses and strains at each layer

    # Getting loads and moments from user input in GUI
    load_list = bx_load.get()
    load_val = load_list.split(',')
    NM = np.array([float(x) for x in load_val])

    # Calculating centerline strains and kappas
    EK = NM @ np.linalg.inv(ABD)
    strain = np.zeros((3, N+1))

    # Calculating strains at top and bottom of each layer 
    for i in range(N+1):
        strain[0,i] = EK[0] + (z[i] * EK[3]) # eps_x
        strain[1,i] = EK[1] + (z[i] * EK[4]) # eps_y
        strain[2,i] = EK[2] + (z[i] * EK[5]) # gamma_xy

    th = np.append(theta, theta[-1])
    
    # Rotating strains back into the fiber frame of reference
    strain_rot = np.zeros((3, N+1))
    for i in range(N+1):
        c = np.cos(th[i] * (np.pi/180))
        s = np.sin(th[i] * (np.pi/180))
        T2 = np.array([[c**2,s**2,c*s],[s**2,c**2,-c*s],[-2*c*s,2*c*s,((c**2)-(s**2))]])
        strain_rot[:,i] = T2 @ strain[:,i]

    # Empty matrices for the top-of-layer stresses and the bottom-of-layer stresses.
    stresstop = np.zeros((3,N))
    stressbot = np.zeros((3,N))
    i,j = 0,0
    while i < N:
        stresstop[:,i] = Qbar[i,:,:] @ strain[:,i]   # stress at top of the layer
        stressbot[:,i] = Qbar[i,:,:] @ strain[:,i+1] # stress at bottom of the layer
        i += 1

    # Rotating the stresses back into the fiber frame of reference.
    stresstop_rot = np.zeros((3, N))
    stressbot_rot = np.zeros((3, N))
    for i in range(N):
        c = np.cos(theta[i] * (np.pi/180))
        s = np.sin(theta[i] * (np.pi/180))
        T = np.array([[c**2,s**2,c*s],[s**2,c**2,-c*s],[-2*c*s,2*c*s,((c**2)-(s**2))]])
        stresstop_rot[:,i] = T @ stresstop[:,i]
        stressbot_rot[:,i] = T @ stressbot[:,i]
    
    # Pulling in strength data from the material data xlsx files
    Slpf = fiber_prop.get('Failure Stress Tension') * 10**6
    Slmf = fiber_prop.get('Failure Stress Compression') * 10**6
    
    Smp = matrix_prop.get('Failure stress tension') * 10**6
    Stm = matrix_prop.get('Failure Stress Compression') * 10**6
    SmLT = matrix_prop.get('Failure Stress Shear') * 10**6

    Slp, Slm, Stp, SLT, elp, elm, etm, etp, eLT = [], [], [], [], [], [], [], [], []
    
    # Calculating the ply max strengths and strains
    for i in range(N):
        ds = np.sqrt((4 * vf[i]) / np.pi) # Square array # Use np.sqrt((2 * np.sqrt(3) * vf[i]) / np.pi) for triangle array
        F = 1 / ((ds * ((EM[mp]/E2F[fp]) - 1)) + 1)
        Fs = 1 / ((ds * ((Gml[i] / G12l[i]) - 1)) + 1)

        Slp.append((Slpf[fp] * vf[i]) + ((EM[mp]*(Slpf[fp]/E1F[fp])) * (1 - vf[i]))) # Equation 4.22, assume fiber fails first
        Stp.append((E2l[i] * Smp[mp]) / (EM[mp] * F)) # Equation 4.36
        etp.append(Stp[i] / E2l[i])
        Slm.append((E1l[i] * etp[i]) / v12l[i]) # Equation 4.33
        # Stm = Stm_[mp]
        SLT.append((G12l[i] * SmLT[mp]) / (Gml[i] * Fs))
        elp.append(Slp[i] / E1l[i])
        elm.append(Slm[i] / E1l[i])
        etm.append(Stm[mp] / E2l[i])
        eLT.append(SLT[i] / G12l[i])

    ## Failure criterion check
    if failcrit_clicked.get() == "Maximum Stress Criterion":
        print('Maximum Stress Criterion')
        for i in range(N):
            if stresstop_rot[0,i] > -Slm[i] and stresstop_rot[0,i] < Slp[i] and stressbot_rot[0,i] > -Slm[i] and stressbot_rot[0,i] < Slp[i] and stresstop_rot[1,i] > -Stm[mp] and stresstop_rot[1,i] < Stp[i] and stressbot_rot[1,i] > -Stm[mp] and stressbot_rot[1,i] < Stp[i] and stresstop_rot[2,i] < SLT[i] and stressbot_rot[2,i] < SLT[i]:
                print('layer ', i+1, ' pass')
            else:
                print('layer ', i+1, ' fail')
    elif failcrit_clicked.get() == 'Maximum Strain Criterion':
        print('Maximum Strain Criterion')
        elp_, elm_, etm_, etp_, eLT_ = np.append(elp, elp[-1]), np.append(elm, elm[-1]), np.append(etm, etm[-1]), np.append(etp, etp[-1]), np.append(eLT, eLT[-1]), 
        for i in range(N+1):
            if strain_rot[0,i] > -elm_[i] and strain_rot[0,i] < elp_[i] and strain_rot[1,i] > -etm_[i] and strain_rot[1,i] < etp_[i] and strain_rot[2,i] < eLT_[i]:
                print('interface ', i+1, ' pass')
            else:
                print('interface ', i+1, ' fail')
    else:
        print('Tsai-Hill Criterion')
        for i in range(N):
            if stresstop_rot[0,i] > 0: # These if-else statements are so the signs of sigma match the signs of
                SL = Slp[i]            # SL and ST as is required for Tsai-Hill criterion.
            else:
                SL = Slm[i]
            if stresstop_rot[1,i] > 0:
                ST = Stp[i]
            else:
                ST = Stm[mp]
            if ((stresstop_rot[0,i]**2 / SL**2) - ((stresstop_rot[0,i] * stresstop_rot[1,i]) / SL**2) + (stresstop_rot[1,i]**2 / ST**2) + (stresstop_rot[2,i]**2 / SLT[i]**2)) < 1:
                print('top of layer    ', i+1, ' pass')
            else:
                print('top of layer    ', i+1, ' fail')
            if stressbot_rot[0,i] > 0:
                SL = Slp[i]
            else:
                SL = Slm[i]
            if stressbot_rot[1,i] > 0:
                ST = Stp[i]
            else:
                ST = Stm[mp]
            if ((stressbot_rot[0,i]**2 / SL**2) - ((stressbot_rot[0,i] * stressbot_rot[1,i]) / SL**2) + (stressbot_rot[1,i]**2 / ST**2) + (stressbot_rot[2,i]**2 / SLT[i]**2)) < 1:
                print('bottom of layer ', i+1, ' pass')
            else:
                print('bottom of layer ', i+1, ' fail')

    a = 2
    b = 4


# Create the Calculate button
calculate_bt = tk.Button(window, text="Calculate", command=Calculate)
calculate_bt.pack()

# Execute tkinter, keeps the GUI open for user input.
window.mainloop()