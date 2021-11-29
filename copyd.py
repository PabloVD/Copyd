#---------------------------
# Disease spread simulator
# Author: Pablo Villanueva Domingo
# Last update: 7/4/20
#---------------------------
import os
import random as rnd
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
import time, datetime
from scipy.special import erf
from scipy.integrate import solve_ivp
from scipy.spatial import cKDTree

time_ini = time.time()

#--- PARAMETERS ---#

# Sizes
dist_ave = 2.#0.5
dist_trip = 10.*dist_ave
trip_prob = 0.5#0.2
Lmax = 100.*np.sqrt(10.)
N_people = 100000
N_inf_in = 10
days = 100

# Virus params
virus_prob = 0.7
virus_dist = 1.
ill_time = 12.#5
ill_sigma = 2.
inc_time = 5.15#3
inc_sigma = 1.3

# Confinement
confinement = 0
pop_conf = 100

# ODEs parameters
solve_ode = 1   # 1 to solve SIR or SEIR equations
beta = 1.5  # between 0.59 and 1.68
sigma = 1./inc_time
gamma = 1./ill_time
tvec = np.linspace(0,days,num=100)

# Others
use_seir = 1    # if 1, employ SEIR population, else employ SIR, no exposition time
sufix = "_N_people_{:d}_Lmax_{:.1f}_virus_prob_{:.1f}_virus_dist_{:.1f}_ill_time_{:.1f}_trip_prob_{:.1f}_confinement_{:d}_use_seir_{:d}".format(N_people,Lmax,virus_prob,virus_dist,ill_time,trip_prob,confinement,use_seir)
col_sus = "blue"
col_exp = "orange"
col_inf = "red"
col_rec = "green"
colors = [col_sus,col_exp,col_inf,col_rec]
labels = ["Susceptible", "Exposed", "Infected", "Recovered"]
ballsize = 2.

#--- CLASSES AND FUNCTIONS ---#

# Person class
class person:

    def __init__(self,x,y,state):
        # Position
        self.x = x
        self.y = y
        # State: 0 susceptible, 1 exposed, 2 infected, 3 recovered
        self.state = state
        self.incub_time = 0
        self.sick_time = 0

    def move(self,dist_ave):
        if rnd.random()<trip_prob:  # long trip
            x_new, y_new = self.x + dist_trip*rnd.uniform(-1.,1.), self.y + dist_trip*rnd.uniform(-1.,1.)
        else:                       # shorter movement
            x_new, y_new = self.x + dist_ave*rnd.uniform(-1.,1.), self.y + dist_ave*rnd.uniform(-1.,1.)
        # Boundary conditions
        if x_new <= 0.: x_new = 0.
        if x_new >= Lmax:   x_new = Lmax
        if y_new <= 0.: y_new = 0.
        if y_new >= Lmax:   y_new = Lmax
        self.x = x_new
        self.y = y_new

# Plot people distribution
def point_plot(pop_list,day):

    fig, ax = plt.subplots()
    for state in range(4):
        pop_pos = np.array([ [guy.x, guy.y] for guy in list_state(pop_list,state) ])
        if pop_pos.size>0:
            ax.scatter(pop_pos[:,0],pop_pos[:,1],color=colors[state],s=ballsize)

    ax.set_xlim([0,Lmax])
    ax.set_ylim([0,Lmax])
    ax.set_title("Day: {}".format(day))
    plt.savefig("daily_plots/day_{:02d}.png".format(day))
    plt.close(fig)

# Distance from one person to another
def distance(x1,y1,x2,y2):
    return np.sqrt( (x1-x2)**2. + (y1-y2)**2. )

# Cumulative probability for incubation and sickness times
def cumul_prob(time,mean_time,sigma):
    return 1./2.*( 1. + erf( (time-mean_time)/np.sqrt(2.)/sigma ) )

# Population of a given state
def list_state(pop_list,state):
    return [guy for guy in pop_list if guy.state == state]

# SEIR evolution equations
def SEIR_evolution_equations(t,y):
    sus, exp, inf = y[0], y[1], y[2]
    sus_eq = -beta/N_people*sus*inf
    exp_eq = beta/N_people*sus*inf - sigma*exp
    inf_eq = sigma*exp - gamma*inf
    return [sus_eq, exp_eq, inf_eq]

#--- INITIALIZATION ---#

# Make some directories if not present yet
for dir in ["daily_plots","summary_plots","outputs"]:
    if not os.path.exists(dir):
        os.system("mkdir "+dir)

pop_list = []

# Distribute people randomly
for i in range(N_people):
    p = person(rnd.uniform(0,Lmax),rnd.uniform(0,Lmax),0)
    pop_list.append(p)

for i in range(N_inf_in):
    patient_zero = rnd.choice(pop_list)
    patient_zero.state = 2


point_plot(pop_list,0)

num_sus = [N_people - N_inf_in]
num_exp = [0]
num_inf = [N_inf_in]
num_rec = [0]


#--- EVOLUTION ---#

for day in range(1,days):

    # If confinement is activated when the number of infected reaches some population pop_conf, probability of doing trips and average distances are lowered
    """if confinement:
        if len(inf_list)>=pop_conf:
            dist = dist_ave/4.
            trip_prob/=4.
        else:
            dist = dist_ave"""

    # Move the people
    for guy in pop_list:
        guy.move(dist_ave)

        # Exposed people become infected people after the exposition time
        if guy.state==1:
            #if guy.incub_time >= inc_time:
            if rnd.random()<cumul_prob(guy.incub_time,inc_time,inc_sigma):
                guy.state = 2
            else:
                guy.incub_time+=1

        # Infected people are recovered after the duration of the sickness
        if guy.state==2:
            #if guy.sick_time >= ill_time:
            if rnd.random()<cumul_prob(guy.sick_time,ill_time,ill_sigma):
                guy.state = 3
            else:
                guy.sick_time+=1

    # Spread the disease
    if len(list_state(pop_list,0))>0 and len(list_state(pop_list,2))>0:

        pos_inf = np.array([ [guy.x, guy.y] for guy in list_state(pop_list,2) ])
        pos_sus = np.array([ [guy.x, guy.y] for guy in list_state(pop_list,0) ])

        tree = cKDTree(pos_sus)
        ball = tree.query_ball_point(pos_inf, virus_dist)
        indexes = []
        for i in range(len(ball)):
            indexes.extend(ball[i])

        for i, guy in enumerate(list_state(pop_list,0)):
            if i in indexes:
                if rnd.random()<virus_prob:
                    if use_seir:
                        guy.state = 1
                    else:
                        guy.state = 2

    num_sus.append(len(list_state(pop_list,0)))
    num_exp.append(len(list_state(pop_list,1)))
    num_inf.append(len(list_state(pop_list,2)))
    num_rec.append(len(list_state(pop_list,3)))

    point_plot(pop_list,day)


#--- SOLVE ODEs ---#

if solve_ode:
    sol = solve_ivp(lambda t, y: SEIR_evolution_equations(t, y), [tvec[0],tvec[-1]], [N_people-N_inf_in, 0, N_inf_in], t_eval = tvec)
    pop_ode = [sol.y[0], sol.y[1], sol.y[2], N_people - (sol.y[0] + sol.y[1] + sol.y[2])]

#--- PLOTS ---#

# Number evolution plot
fig, ax = plt.subplots()
ax.set_xlabel("Days")
ax.set_ylabel("Population")
#ax.set_yscale("log")
customlegend = []
num_pop = [num_sus, num_exp, num_inf, num_rec]
for i in range(4):
    ax.plot(range(days),np.array(num_pop[i])/N_people,color=colors[i],linestyle="-")
    if solve_ode:   ax.plot(tvec,np.array(pop_ode[i])/N_people,color=colors[i],linestyle=":")
    customlegend.append(Line2D([0], [0], color=colors[i], label=labels[i]))

if solve_ode:
    customlegend.append(Line2D([0], [0], color="k", linestyle="-", label="N-body"))
    customlegend.append(Line2D([0], [0], color="k", linestyle=":", label="SEIR ODEs"))

ax.legend(handles=customlegend)
plt.savefig("summary_plots/number_evolution"+sufix+".png", bbox_inches='tight', dpi=300)
plt.close(fig)
np.savetxt("outputs/number_evolution_table"+sufix+".dat",np.transpose([np.array(range(days)),np.array(num_sus),np.array(num_exp),np.array(num_inf),np.array(num_rec)]))

# Gif
images = []
plotnames = sorted(glob.glob("daily_plots/day*"))

img, *imgs = [Image.open(f) for f in plotnames]
img.save(fp="summary_plots/evolution"+sufix+".gif", format='GIF', append_images=imgs, save_all=True, duration=200, loop=0)

print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
