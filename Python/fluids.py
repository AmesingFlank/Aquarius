import numpy as np
import random
import taichi as ti

ti.init(arch=ti.cuda)



grid_size = (64,64)
(grid_size_x,grid_size_y) = grid_size

window_size = (512,512)
(window_size_x,window_size_y) = window_size

cell_size = 10 / grid_size_x

ppc = 8

epsilon = cell_size * 0.001

g = 9.8
dt = 0.016

FLUID = 1
EMPTY = 0

# from mgpcg_advanced import MGPCG  # examples/mgpcg_advanced.py
# mgpcg = MGPCG(dim=2, N=grid_size_x, n_mg_levels=6)

@ti.func
def cell_index(x,y):
    return x*grid_size_y + y

@ti.func
def lerp(a,b,t):
    return a * (1.0-t) + b * (t)

@ti.func
def kernel_1d(r,support):
    r = abs(r/support)
    # result = 0.0
    # if r < 0.5:
    #     result = 0.75 - r*r
    # if 0.5 <= r and r <= 1.5:
    #     result = 0.5* ((1.5-r) ** 2)
    # return result
    return max(0,1-r)
    

@ti.func
def kernel_2d(r,support):
    return kernel_1d(r[0],support) * kernel_1d(r[1],support)



@ti.data_oriented
class Grid:
    def __init__(self):
        self.velocity = ti.Vector.field(2,dtype=ti.f32,shape = (grid_size_x+1,grid_size_y+1))
        self.velocity_weights = ti.Vector.field(2,dtype=ti.f32,shape = (grid_size_x+1,grid_size_y+1))
        self.content = ti.field(dtype=ti.i32,shape = (grid_size_x,grid_size_y))
        self.particle_count = ti.field(dtype=ti.i32,shape = (grid_size_x,grid_size_y))

        self.divergence = ti.field(dtype=ti.f32,shape = (grid_size_x,grid_size_y))
        self.pressure = ti.field(dtype=ti.f32,shape = (grid_size_x,grid_size_y))

    @ti.kernel
    def init_grid(self):
        for I in ti.grouped(grid.velocity):
            grid.velocity[I] = ti.Vector([0,0])
            grid.velocity_weights[I] = ti.Vector([0,0])
    
    @ti.func
    def get_pressure(self,i,j):
        i = max(0,min(grid_size_x-1,i))
        j = max(0,min(grid_size_y-1,j))
        return self.pressure[i,j]

    @ti.func
    def get_velocity(self,pos):
        pos[0] = max(epsilon,min(grid_size_x*cell_size-epsilon,pos[0]))
        pos[1] = max(epsilon,min(grid_size_y*cell_size-epsilon,pos[1]))

        x = ti.cast(ti.floor(pos[0] / cell_size),ti.i32)
        y = ti.cast(ti.floor(pos[1] / cell_size),ti.i32)

        vel = ti.Vector([0.0,0.0])
        sum_weights = ti.Vector([0.0,0.0])

        for i,j in ti.ndrange(3,3):
            this_x = x+i-1
            this_y = y+j-1

            if this_x < 0 or this_x >= grid_size_x+1 or this_y < 0 or this_y >= grid_size_y+1:
                continue

            x_vel_pos = ti.Vector([this_x,this_y+0.5]) * cell_size
            y_vel_pos = ti.Vector([this_x+0.5,this_y]) * cell_size

            x_weight = kernel_2d(x_vel_pos-pos,cell_size)
            y_weight = kernel_2d(y_vel_pos-pos,cell_size)

            weights = ti.Vector([x_weight,y_weight])
            velocity_contribution = self.velocity[this_x,this_y] * weights

            vel += velocity_contribution
            sum_weights += weights

        if sum_weights[0] > 0:
            vel[0] /= sum_weights[0]
        if sum_weights[1] > 0:
            vel[1] /= sum_weights[1]
            
        return vel

@ti.data_oriented
class Particles:
    def __init__(self):
        self.count = 0
        self.position_host = []
        self.init_host_particles()
        
    def init_host_particles(self):
        for x,y in ti.ndrange(int(grid_size_x/2),int(grid_size_y/2)):
            for i in range(ppc):
                x_pos = (x+ random.random())*cell_size + grid_size_x*cell_size/4
                y_pos = (y+ random.random())*cell_size + grid_size_x*cell_size/4
                x_pos = max(epsilon,min(cell_size*grid_size_x-epsilon,x_pos))
                y_pos = max(epsilon,min(cell_size*grid_size_y-epsilon,y_pos))
                self.position_host.append([x_pos,y_pos])
        self.count = len(self.position_host)
        self.position_host = np.array(self.position_host)

        self.position = ti.Vector.field(2,dtype=ti.f32,shape = self.count)
        self.velocity = ti.Vector.field(2,dtype=ti.f32,shape = self.count)
    
    def init_particles(self):
        self.position.from_numpy(self.position_host)
        self.init_velocity()

    @ti.kernel
    def init_velocity(self):
        for v in self.velocity:
            self.velocity[v]=ti.Vector([0,0])



@ti.data_oriented
class Renderer:
    def __init__(self):
        self.pixels = ti.Vector.field(3,dtype=ti.f32, shape=window_size)
        #self.init_pixels()

    @ti.kernel
    def clear_pixels(self):
        for p in ti.grouped(self.pixels):
            self.pixels[p] = ti.Vector([0.9,0.9,0.9])

    @ti.kernel
    def render_particles(self):
        for p in particles.position:
            x_pos = particles.position[p][0]
            y_pos = particles.position[p][1]
            x_pos = x_pos * window_size_x / (grid_size_x * cell_size)
            y_pos = y_pos * window_size_y / (grid_size_y * cell_size)

            x = ti.cast(x_pos,ti.i32)
            y = ti.cast(y_pos,ti.i32)
            self.pixels[x,y] = ti.Vector([0,0,1])

particles = Particles()
renderer = Renderer()
grid = Grid()


@ti.kernel
def grid_to_particles():
    for p in particles.velocity:
        pos = particles.position[p]
        v = grid.get_velocity(pos)
        particles.velocity[p] = v
            
    

@ti.kernel
def move_particles():
    for p in particles.velocity:
        pos = particles.position[p]
        
        u1 = grid.get_velocity(pos)
        u2 = grid.get_velocity(pos+dt * u1 / 2)
        u3 = grid.get_velocity(pos+dt * u2 *3 / 4)

        final_pos = pos + dt *  (u1 * 2 / 9 + u2 * 3 / 9 + u3 * 4 / 9)
        final_pos = pos + dt * particles.velocity[p]

        bounce = 0

        if final_pos[0] < epsilon:
            final_pos[0] = epsilon
            particles.velocity[p][0] *= bounce
        elif final_pos[0] > cell_size*grid_size_x - epsilon:
            final_pos[0] = cell_size*grid_size_x - epsilon
            particles.velocity[p][0] *= bounce

        if final_pos[1] < epsilon:
            final_pos[1] = epsilon
            particles.velocity[p][1] *= bounce
        elif final_pos[1] > cell_size*grid_size_y - epsilon:
            final_pos[1] = cell_size*grid_size_y - epsilon
            particles.velocity[p][1] *= bounce

        particles.position[p] = final_pos



@ti.kernel
def particles_to_grid():
    for I in ti.grouped(grid.velocity):
        grid.velocity[I] = ti.Vector([0,0])
        grid.velocity_weights[I] = ti.Vector([0,0])
    for I in ti.grouped(grid.content):
        grid.content[I] = EMPTY
    for I in ti.grouped(grid.particle_count):
        grid.particle_count[I] = 0

    for p in particles.velocity:
        pos = particles.position[p]
        vel = particles.velocity[p]
        x = ti.cast(ti.floor(pos[0] / cell_size),ti.i32)
        y = ti.cast(ti.floor(pos[1] / cell_size),ti.i32)

        if grid.content[x,y] == EMPTY:
            grid.content[x,y] = FLUID
        
        grid.particle_count[x,y] += 1

        for i,j in ti.ndrange(3,3):
            this_x = x+i-1
            this_y = y+j-1

            if this_x < 0 or this_x >= grid_size_x+1 or this_y < 0 or this_y >= grid_size_y+1:
                continue

            x_vel_pos = ti.Vector([this_x,this_y+0.5]) * cell_size
            y_vel_pos = ti.Vector([this_x+0.5,this_y]) * cell_size

            x_weight = kernel_2d(x_vel_pos-pos,cell_size)
            y_weight = kernel_2d(y_vel_pos-pos,cell_size)

            weights = ti.Vector([x_weight,y_weight])
            velocity_contribution = vel * weights

            grid.velocity[this_x,this_y] += velocity_contribution
            grid.velocity_weights[this_x,this_y] += weights
    
    for I in ti.grouped(grid.velocity):
        if grid.velocity_weights[I][0] > 0:
            grid.velocity[I][0] = grid.velocity[I][0] / grid.velocity_weights[I][0]
        else:
            grid.velocity[I][0] = 0
            
        if grid.velocity_weights[I][1] > 0:
            grid.velocity[I][1] = grid.velocity[I][1] / grid.velocity_weights[I][1]
        else:
            grid.velocity[I][1] = 0
        
    for I in ti.grouped(grid.content):
        if grid.content[I] != FLUID:
            grid.pressure[I] = 0
        
@ti.kernel
def apply_gravity():
    for x,y in ti.ndrange(grid_size_x,grid_size_y):
        if grid.content[x,y] == FLUID or (y-1>=0 and grid.content[x,y-1] == FLUID):
            grid.velocity[x,y][1] =  grid.velocity[x,y][1]-dt*g

@ti.kernel
def apply_boundaries():
    for x in range(grid_size_x):
        grid.velocity[x,0][1] = 0
        grid.velocity[x,grid_size_y][1] = 0
    
    for y in range(grid_size_y):
        grid.velocity[0,y][0] = 0
        grid.velocity[grid_size_x,y][0] = 0

@ti.kernel
def compute_divergence():
    for x,y in ti.ndrange(grid_size_x,grid_size_y):
        if grid.content[x,y]==FLUID:
            div = grid.velocity[x+1,y][0] - grid.velocity[x,y][0] + grid.velocity[x,y+1][1] - grid.velocity[x,y][1]
            #div -= (grid.particle_count[x,y] - ppc) * 0.001
            grid.divergence[x,y] = div
        else:
            grid.divergence[x,y] = 0

        

@ti.kernel
def jacobi_iter():
    for x,y in ti.ndrange(grid_size_x,grid_size_y):
        if grid.content[x,y] != FLUID:
            grid.pressure[x,y] = 0
        else:
            div = grid.divergence[x,y]
            RHS = -div

            pressure = 0.0
            pressure += grid.get_pressure(x-1,y)
            pressure += grid.get_pressure(x+1,y)
            pressure += grid.get_pressure(x,y-1)
            pressure += grid.get_pressure(x,y+1)
            pressure += RHS
            pressure /= 4.0
            grid.pressure[x,y] = pressure

def solve_poisson_jacobi():
    for i in range(200):
        jacobi_iter()

# def solve_poisson_mgpcg():
#     mgpcg.init(grid.divergence,-1)
#     mgpcg.solve(max_iters=10)
#     mgpcg.get_result(grid.pressure)

@ti.kernel
def apply_pressure():
    for x,y in ti.ndrange(grid_size_x,grid_size_y):
        if x>0 and (grid.content[x,y] == FLUID or (x-1>=0 and grid.content[x-1,y] == FLUID)):
            grid.velocity[x,y][0] = grid.velocity[x,y][0] - (grid.pressure[x,y]-grid.pressure[x-1,y])
        if y>0 and (grid.content[x,y] == FLUID or (y-1>=0 and grid.content[x,y-1] == FLUID)):
            grid.velocity[x,y][1] = grid.velocity[x,y][1] - (grid.pressure[x,y]-grid.pressure[x,y-1])

@ti.kernel
def extrapolate_velocity_once():
    for x,y in ti.ndrange(grid_size_x+1,grid_size_y+1):
        vel = grid.velocity[x,y]
        if vel[0] > 0:
            if x+1 < grid_size_x and grid.content[x+1,y] != FLUID:
                grid.velocity[x+1,y][0] = vel[0]
        if vel[0] < 0:
            if x-1 > 0 and grid.content[x-1,y] != FLUID:
                grid.velocity[x-1,y][0] = vel[0]
        if vel[1] > 0:
            if y+1 < grid_size_y and grid.content[x,y+1] != FLUID:
                grid.velocity[x,y+1][1] = vel[1]
        if vel[1] < 0:
            if y-1 > 0 and grid.content[x,y-1] != FLUID:
                grid.velocity[x,y-1][1] = vel[1]
                

def extrapolate_velocity():
    for i in range(10):
        extrapolate_velocity_once()
        

def substep():
    grid_to_particles()
    move_particles()
    particles_to_grid()

    apply_gravity()
    apply_boundaries()

    compute_divergence()
    solve_poisson_jacobi()
    apply_pressure()
    apply_boundaries()


    #extrapolate_velocity()
    
    

particles.init_particles()
grid.init_grid()

gui = ti.GUI('Aquarius', window_size)
while gui.running:
    substep()
    gui.clear(0xFFFFFF)
    renderer.clear_pixels()
    renderer.render_particles()
    #gui.circles(particles.position.to_numpy(),radius = 5,color=0x0000FF)
    gui.set_image(renderer.pixels)
    gui.show()
 