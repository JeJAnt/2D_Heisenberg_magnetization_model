import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from math import *
from numpy.random import default_rng
import copy
from time import time

def print_progress(step, total):
    #taken from exercise ising.py
    """
    Prints a progress bar.
    
    Args:
        step (int): progress counter
        total (int): counter at completion
    """

    message = "simulation progress ["
    total_bar_length = 60
    percentage = int(step / total * 100)
    bar_fill = int(step / total * total_bar_length)
    for i in range(total_bar_length):
        if i < bar_fill:
            message += "|"
        else:
            message += " "
    
    message += "] "+str(percentage)+" %"
    if step < total:
        print(message, end="\r")     
    else:
        print(message) 

# functions for animation: update_quiver, update_colormesh and animate_grid

def update_quiver(frame, history, quiver):
    # Update the quiver plot for 2D vector animation
    spins = history[frame]
    # vector direction
    U = np.cos(spins)
    V = np.sin(spins)
    # Normalize angles to [0, 1] for colormap
    colors = spins / (2 * np.pi)  
    # change orientation and color of vectors
    quiver.set_UVC(U, V, colors)
    #quiver.set_UVC(U,V)
    # return a tuple containing quiver object
    return quiver,

def update_colormesh(frame, history, colormesh):
    # update coloremesh animation
    spins = history[frame]
    # Normalize angles to [0, 1] for colormap
    colors = spins / (2 * np.pi)  
    # Update colormesh with new colors
    colormesh.set_array(colors.ravel())  
    # return a tuple containing colormesh object
    return colormesh,

def animate_grid(history):
    # function for animation
    # either vector or colormesh
    # parameters: list, history: list of grids from the metropolis algorithm

    # Lattice size
    L = history[0].shape[0]  
    # Number of frames
    n_frames = len(history)

    print("animating "+str(n_frames)+" frames")

    # Set up the plot
    fig, ax = plt.subplots()

    ################# vector animation
    # set the position of vectors
    x = np.arange(L)
    y = np.arange(L)
    X, Y = np.meshgrid(x, y)
    # set the direction of vectors
    U = np.cos(history[0])
    V = np.sin(history[0])
    #set vector colors according to angle

    # Normalize angles to [0, 1] for colormap
    colors = history[0] / (2 * np.pi)  
    # plot a 2D field of arrows with quiver
    quiver = ax.quiver(X, Y, U, V, colors, cmap = 'hsv')


    #quiver = ax.quiver(X, Y, U, V)

    # animation command
    anim = ani.FuncAnimation(fig, update_quiver, fargs=(history, quiver), frames=n_frames, interval = 50)
    plt.title('2D XY Model Spin Configuration')
    # save as gif
    anim.save('xy_vector_animation_'+str(L)+'.gif', writer='pillow')
    plt.show()

    #################### alternative animation for colormesh representation

    ''' # set colormesh according to vector angles
    colors = history[0] / (2 * np.pi) 
    colormesh = ax.pcolormesh(colors, cmap='hsv', shading='auto')
    #animation command
    anim = ani.FuncAnimation(fig, update_colormesh, fargs=(history, colormesh), frames=n_frames, interval=50)
    plt.title('2D XY model spin configuration colormesh representation')
    plt.colorbar(colormesh, ax=ax)
    # save as gif
    anim.save('xy_colormesh_animation_'+str(L)+'.gif', writer='pillow')
    plt.show()'''

def generate_grid(lattice_size,all_up = False):
    # generate grid as NxN array
    # parameters: 
    # int, lattice_size: size of lattice
    # bool, all_up: whether initial spins parallel or random
    # return: array, grid: LxL grid of vector angles
    if all_up:
        grid = np.full((lattice_size,lattice_size), 2*np.pi)
        return grid
    else:
        # each lattice point is a spin unit vector angle between [0,2pi]
        grid = np.random.rand(lattice_size, lattice_size) * 2 * np.pi
        return grid




def calculate_energy_change(grid,i,j,delta_theta,J=1):
    # check if spinning some vector makes the local energy decrease
    # parameters:
    # array, grid: lattice shape with unit vector angles as values
    # int, i & j: i:th and j:th point in lattice
    # float, delta_theta: the new angle for which we calculate energy change
    # int, J: coupling constant
    # return: float, delta-energy: change in energy from the angle change

    L = grid.shape[0]
    delta_energy = 0.0
    current_angle = grid[i,j]
    new_angle = current_angle + delta_theta


    # Neighbors of the spin (i, j)
    neighbors = [
        ((i + 1) % L, j),
        ((i - 1) % L, j),
        (i, (j + 1) % L),
        (i, (j - 1) % L)]
    

    # Calculate the change in energy due to the rotation
    for ni, nj in neighbors:
        delta_energy -= J * (np.cos(new_angle - grid[ni, nj]) - np.cos(grid[i,j] - grid[ni, nj]))

    return delta_energy

def calculate_magnetic_moment(grid):
    # sum each spin cosine and sine components
    # calculate magnitude of vector M and normalize by N
    #parameters: array, grid: lattice shape with unit vector angles as values
    # return: float, M: magnetization in grid

    # function computes the x and y components of the magnetization by summing the sine and cosine components of each spin
    L = grid.shape[0]
    Mx = np.sum(np.cos(grid))
    My = np.sum(np.sin(grid))
    # total magnetization is normalized by total number of spins
    M = np.sqrt(Mx**2 + My**2) / (L * L)
    return M

def calculate_statistics(samples,grid):
    # parameter: array, samples: array for which to calculate statistics
    # array, grid: lattice shape with unit vector angles as values
    #return: float: mean, float: variance, float: std, float: standard error of mean

    # at the end didn't use variance and std in the study but left them implemented anyway

    # estimation of mean
    mean = sum(samples)/len(samples)

    L = len(grid[0])

    #variance estimation by sample variation
    diff = []
    for i in samples:
        diff.append((i-mean)**2)
        variance = sum(diff) / (L-1)

    # standard deviation
    std = sqrt(variance)

    #standard error of mean
    error = std / sqrt(L)

    return mean, variance, std, error


def metropolis_step(grid,temperature,delta_theta_max,J=1):
    # function for generating new angles and testing them

    #parameters:
    # array, grid: lattice shape with unit vector angles as values
    # float, T: temperature
    # float, delta_theta_max: maximum value for possible angle change
    # int, J: coupling constant

    lattice_size = len(grid)
    coordinates = random.integers(0,lattice_size,2)
    i = coordinates[0]
    j = coordinates[1]

    # generate random number for evaluating if spin rotation is accepted
    rng = random.random()

    # create some new angle to be tested randomly from interval 
    delta_theta = np.random.uniform(-delta_theta_max, delta_theta_max)

    #calculate would be energy change if spin was to be rotated
    spin_energy = calculate_energy_change(grid,i,j,delta_theta)

    # Boltzmann probability
    boltzmann_prob = np.exp(-spin_energy / temperature)
    
    if spin_energy < 0:
        grid[i,j] += delta_theta
        #modulo 2pi
        grid[i, j] %= 2 * np.pi 
    else:
        if rng < boltzmann_prob:
            grid[i,j] += delta_theta
            # modulo 2pi
            grid[i, j] %= 2 * np.pi 


def run_metropolis_algorithm(grid,temperature,delta_theta_max,n_steps,therm_time, animation = True):
    #
    # inspiration for algorithm structure taken from exercise ising.py
    #
    #parameters:
    # grid, array: lattice shape with unit vector angles as values
    # temperature, float: temperature
    # n_steps, int: number of simulation steps
    # therm_time, int: how many steps to take before recording starts
    # animation, bool: button for animation
    # return: array of magnetizations

    # keep record of simulation time
    start_time = time()

    # store moments in list
    moments = []
    # keep a record of the grid for animation
    history = []
    # number of spins in grid
    spins = len(grid)* len(grid)

    for step in range(n_steps+1):
        print_progress(step,n_steps)

        #for each step, change spins as many times as there are spins total
        for _ in range(spins):
            metropolis_step(grid,temperature,delta_theta_max)


        #start recording after thermalization period
        # also check that recording starts from step that is divisible by 10 to avoid correlated states

        if step > therm_time and step % 10 == 0:
            moment = calculate_magnetic_moment(grid)
            moments.append(abs(moment))

        if animation:
            history.append(copy.deepcopy(grid))
    

    print_progress(n_steps, n_steps)
    end_time = time()
    print("simulation took "+str(end_time-start_time)+" s")
    if animation:
        animate_grid(history)
    
    
    return np.array(moments)


def run_temperature_series(grid,temperatures,delta_theta_max,n_steps,therm_steps):
    #
    # parameters:
    # grid,  array: lattice shape with unit vector angles as values
    # temperatures, array, list of temperatures
    # int, n_steps: number of steps to do in the metropolis algorithm
    # int, therm_steps: number of steps to be taken before data collection
    # return: array, array, mag_means: means of magnetizations for all temperatures, mag_errors: mean of mag errors for all temperatures
    
    #
    mag_means = []
    mag_errors = []
    #run metropolis for every T in temperatures

    
    start_time = time()

    for T in temperatures:
        print(f'starting simulation at T = {T}')
        magnetization = run_metropolis_algorithm(grid,T,delta_theta_max,n_steps,therm_steps,animation= False)
        #calculate statistics from each simulation run
        average, variance, standr_dev, standr_error =  calculate_statistics(magnetization,grid)
        mag_means.append(average)
        mag_errors.append(standr_error)
    end_time = time()
    print('temperature series total simulation time:'+str(end_time-start_time)+'s')
    return np.array(mag_means), np.array(mag_errors)


def main(temp_series = False):

    # main program
    # parameter: bool, temp_series: if False, single metropolis algorithm is done, else temperature series is done

    # global parameters:
    ##########################################################################################
    # grid size
    lattice_size = 20
    # simulation 'time'
    n_steps = 5000
    # amount of thermalization 'time' before statistics collection
    therm_steps = 4000
    # Maximum angle for a single spin move                                                  
    delta_theta_max = np.pi / 20
    # make initially spins parallel or random                                               
    all_up = True
    # create grid                                                                           
    grid = generate_grid(lattice_size, all_up = all_up)                                     
    #########################################################################################
    
    if temp_series == False:
        ###########run single metropolis algorithm

        # single simulation parameters:
        # temperature
        temperature = 1.2
        # command for animation
        animate = True

        print(f'Starting single temperature simulation with parameters: \n lattice size: {lattice_size}\n temperature: {temperature} \n simulation steps: {n_steps}')
        # collect magnetizations with metropolis algorithm
        magnetizations = run_metropolis_algorithm(grid,
                                                temperature,
                                                delta_theta_max,
                                                n_steps,
                                                therm_steps,
                                                animate)
        
        # calculate statistics
        mean, variance, std, error = calculate_statistics(magnetizations,grid)

        # plot magnetization
        plt.plot(magnetizations)
        # plot calculated average
        plt.plot(np.linspace(0,
                            len(magnetizations), 
                            len(magnetizations)),
                            mean*np.ones(len(magnetizations)),
                            'b:')
        # plot confidence interval
        plt.fill_between(np.linspace(0,
                                    len(magnetizations),
                                    len(magnetizations)),
                                    (mean-2*error)*np.ones(len(magnetizations)),
                                    (mean+2*error)*np.ones(len(magnetizations)),
                                    color = 'b', alpha=0.1 )
        # label axis
        plt.xlabel("simulation step")
        plt.ylabel("magnetization")
        plt.savefig('xy_magnetizations_'+str(lattice_size)+'_'+str(temperature)+'.png')
        plt.show()

        
    else:
        ############ run temperature series algorithm

        # temperature series parameters:
        # minimum temperature
        T_min = 0.1
        # maximum temperature
        T_max = 1.0
        # number of steps between min and max temperatures
        T_steps = 10
        #
        temperatures = np.linspace(T_min,T_max,T_steps)

        print(f'Starting temperature series simulation with parameters: \n lattice size: {lattice_size}\n temperatures: {T_min}-{T_max} \n simulation steps: {n_steps}')

        # collect magnetization and error values with temperature series
        mag_series, error_series = run_temperature_series(grid,
                                                temperatures,
                                                delta_theta_max,
                                                n_steps,
                                                therm_steps)
        

        # plot temperature series magnetizations with error bars
        #print(temperatures)
        #print(mag_series)

        plt.errorbar(temperatures,
                    mag_series,
                    yerr= error_series,
                    fmt='o',
                    capsize=10)
        
        plt.plot()
        plt.xlabel("temperature")
        plt.ylabel("magnetization")
        plt.savefig('temperature_series_plot_'+str(lattice_size)+'_['+str(T_min)+','+str(T_max)+'].png')
        plt.show()
        


    # implemented a lattice series but didn't have time to get proper results
    # commented out for user simplicity 
    # with n_steps = 1000 program took 52 minutes
    #################################### lattice series + time series
    '''
    #lattice sizes
    lattice_sizes = [20,30,40]
    T_min = 0.1
    # maximum temperature
    T_max = 2.0
    # number of steps between min and max temperatures
    T_steps = 20
    temperatures = np.linspace(T_min,T_max,T_steps)

    # dict for results for each lattice
    results = {L: [] for L in lattice_sizes}
    start_time = time()
    for L in lattice_sizes:
        grid = generate_grid(L)
        print('Starting simulation with lattice size: '+str(L))
        for T in temperatures:
            print(f'starting simulation at T = {T}')
            magnetization = run_metropolis_algorithm(grid,T,delta_theta_max,n_steps,therm_steps,animation= False)
            #print('magnetization: '+str(magnetization))
            mag_ave, var_ave, std_ave, error_ave  = calculate_statistics(magnetization)
            #print('mag_ave: '+str(mag_ave))
            results[L].append(mag_ave)
    end_time = time()
    print('lattice x temp simulation took '+str(end_time-start_time)+' s')
    for L in lattice_sizes:
        plt.plot(temperatures, results[L], label=f"L={L}")
    plt.xlabel('Temperature')
    plt.ylabel('magnetizations')
    plt.legend()
    plt.title('Magnetizations vs Temperature for different Lattice Sizes')
    plt.show()
    '''
    #######################


if __name__ == "__main__":
    random = default_rng()
    main(temp_series=True)
else:
    random = default_rng()

# simulation instructions:
# main: 
#   run either single metropolis simulation with temp_series = False or temperature series with True
#
# global parameters found at the beginning of main
# simulation specific parameters found at the beginning of if-else statements
#
# in function animate_grid vector animation done as default but option for colormesh animation left commented out - works
# at the end of main left commented out an option for multiple lattice sizes simulation  - not good results