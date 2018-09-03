# Copied and adapted from https://is.gd/S53PwY

from manta import *
import os, math
import numpy as np
# DELETE BEGIN
# import shutil, sys, time, inspect
# sys.path.append("../")
#import paramhelpers as ph

# def get_mean(grid, dim=[64,64,64]):
#     m = 0
#     for i in range(dim[0]):
#         for j in range(dim[1]):
#             for k in range(dim[2]):
#                 m += grid.index(i,j,k)
#     return m/(64**3)
# DELETE END

# Main parameters  -----------------------------------------------------------------#
steps = 200
save_data = True
save_ppm = False
sim_nr = 1  # start ID
show_gui = False
base_path = './'
np_seed = -1
max_saves = 100

# Enable for debugging.
#steps = 30       # shorter test
#save_data = False # debug, dont write...
#show_gui  = True  # show UI

# Scene settings  ------------------------------------------------------------------#
setDebugLevel(1)

# Solver parameters  ---------------------------------------------------------------#
dim = 3
res = 64
offset = 20 # Start saving frames after so much time steps.
interval = 1 # How often frames are saved.

gs = vec3(res, res, 1 if dim==2 else res)
buoy = vec3(0, -1e-3, 0) # Add external force.

# Create main solver object.
sm = Solver(name='main', gridSize=gs, dim=dim)
sm.timestep = .5

# DELETE START
# timings = Timings()
# DELETE END

# Simulation Grids  ----------------------------------------------------------------#
flags = sm.create(FlagGrid)
velocity = sm.create(MACGrid)
density = sm.create(RealGrid)
pressure = sm.create(RealGrid)

# Open boundaries.
b_width = 1
flags.initDomain(boundaryWidth=b_width)
flags.fillGrid()

setOpenBound(flags, b_width, 'yY', FlagOutflow|FlagEmpty) 

# Inflow sources -------------------------------------------------------------------#
if np_seed > 0:
    np.random.seed(np_seed)

# Initialize random inflow sources.
noise_nr = 20
noise = []
sources = []
center_pos = vec3(.5, .3, .5)
seeds = np.random.randint(1e4, size=noise_nr)
randoms = np.random.rand(noise_nr, 8)

for n in range(noise_nr):
    noise.append(sm.create(NoiseField, fixedSeed=int(seeds[n])))
    noise[n].posScale = vec3(res*.1*(randoms[n][7] + 1))
    noise[n].clamp = True
    noise[n].clampNeg = 0
    noise[n].clampPos = 1.
    noise[n].valScale = 5.
    noise[n].valOffset = -.01 # some gap
    noise[n].timeAnim = .3
    noise[n].posOffset = vec3(1.5)
	
    # Random center offsets.
    center_off = vec3(.4) * (vec3(randoms[n][0], randoms[n][1], randoms[n][2])-vec3(.5))
    radius_rand = .035 + .035*randoms[n][3]
    upscale = vec3(.95) + vec3(.1)*vec3(randoms[n][4], randoms[n][5], randoms[n][6])

    if dim == 2: 
        center_off.z = 0.
        upscale.z = 1.
    sources.append(sm.create(Sphere, center=gs*(center_pos+center_off), radius=gs.x*radius_rand, scale=upscale)) 
    densityInflow(flags=flags, density=density, noise=noise[n], shape=sources[n], scale=3., sigma=2.)

# Inititalize random velocity.
velocity_rand = np.random.rand(3)
v1pos = vec3(.7 + .4*(velocity_rand[0]-.5)) # In (.5, .9).
v2pos = vec3(.3 + .4*(velocity_rand[1]-.5)) # In (.1, .5).
vtheta = velocity_rand[2]*math.pi*.5
velInflow = .04*vec3(math.sin(vtheta), math.cos(vtheta), 0)

if dim == 2:
    v1pos.z = v2pos.z = .5
    sourceV1 = sm.create(Sphere, center=gs*v1pos, radius=gs.x*.1, scale=vec3(1))
    sourceV2 = sm.create(Sphere, center=gs*v2pos, radius=gs.x*.1, scale=vec3(1))
    sourceV1.applyToGrid(grid=velocity, value=-velInflow*float(gs.x))
    sourceV2.applyToGrid(grid=velocity, value=velInflow*float(gs.x))
elif dim == 3:
    velocity_rand_more = np.random.rand(3)
    vtheta2 = velocity_rand_more[0]*math.pi*.5
    vtheta3 = velocity_rand_more[1]*math.pi*.5
    vtheta4 = velocity_rand_more[2]*math.pi*.5
    for dz in range(1, 10):
        v1pos.z = v2pos.z = (.1*dz)
        vtheta_xy = vtheta * (1.-.1*dz) + vtheta2*.1*dz
        vtheta_z  = vtheta3 * (1.-.1*dz) + vtheta4*.1*dz
        velInflow = .04*vec3(math.cos(vtheta_z)*math.sin(vtheta_xy), math.cos(vtheta_z)*math.cos(vtheta_xy), math.sin(vtheta_z))
        sourceV1 = sm.create(Sphere, center=gs*v1pos, radius=gs.x*.1, scale=vec3(1))
        sourceV2 = sm.create(Sphere, center=gs*v2pos, radius=gs.x*.1, scale=vec3(1))
        sourceV1.applyToGrid(grid=velocity, value=-velInflow*float(gs.x))
        sourceV2.applyToGrid(grid=velocity, value=velInflow*float(gs.x))

# Setup UI -------------------------------------------------------------------------#
if show_gui and GUI:
    gui = Gui()
    gui.show()
    # gui.pause()

t = 0

if save_data:
    folder_nr = sim_nr
    pathaddition = 'sim_%04d/' % folder_nr
    while os.path.exists(base_path + pathaddition):
        folder_nr += 1
        pathaddition = 'sim_%04d/' % folder_nr

    simPath = base_path + pathaddition
    print("Using output dir '%s'" % (simPath)) 
    sim_nr = folder_nr
    os.makedirs(simPath)

# Main loop ------------------------------------------------------------------------#
saved_images = 0
saved_once = False
while t < steps+offset and saved_images < max_saves:
    mantaMsg("Current time t: " + str(t*sm.timestep))
    
    advectSemiLagrange(flags=flags, vel=velocity, grid=density, order=2, openBounds=True, boundaryWidth=b_width)
    advectSemiLagrange(flags=flags, vel=velocity, grid=velocity, order=2, openBounds=True, boundaryWidth=b_width)
    setWallBcs(flags=flags, vel=velocity)
    addBuoyancy(density=density, vel=velocity, gravity=buoy, flags=flags)

    if t < offset: 
        vorticityConfinement(vel=velocity, flags=flags, strength=.05)

    solvePressure(flags=flags, vel=velocity, pressure=pressure, cgMaxIterFac=10., cgAccuracy=.0001)
    setWallBcs(flags=flags, vel=velocity)

    # Save data.
    if save_data and t >= offset and (t-offset) % interval == 0:
        tf = (t-offset) / interval
        mean = density.getL1() / res**dim
        print("Mean: {}".format(mean))
        if mean > .02 or saved_once:
            if mean < .02:
                raise ValueError("Mean too small!")
            density.save(simPath + 'density_%04d.uni' % (saved_images))
            velocity.save(simPath + 'velocity_%04d.uni' % (saved_images))
            saved_once = True
            if save_ppm:
                projectPpmFull(density, simPath + 'density_%04d_%04d.ppm' % (sim_nr, saved_images), 0, 1.)
            saved_images += 1

    sm.step()
    t += 1
