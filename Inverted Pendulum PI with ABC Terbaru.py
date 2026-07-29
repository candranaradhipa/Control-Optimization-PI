import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameter sistem pendulum terbalik
mass_pendulum = 0.5  # Massa pendulum (kg)
mass_wheel = 0.2     # Massa roda (kg)
length_pendulum = 0.3  # Panjang pendulum (m)
gravity = 9.81       # Gravitasi (m/s^2)
inertia = 0.006      # Inersia pendulum (kg.m^2)
damping = 0.05       # Penurunan redaman untuk gerakan lebih halus

# Waktu simulasi
dt = 0.01  # Interval waktu (s)
time = np.arange(0, 10, dt)  # Rentang waktu (10 detik)

# Fungsi simulasi pendulum terbalik dengan gangguan awal
def simulate_pendulum(Kp, Ki, disturbance=0.0):
    theta = np.zeros_like(time)  # Sudut pendulum (rad)
    theta_dot = np.zeros_like(time)  # Kecepatan sudut (rad/s)
    integral_error = 0
    desired_theta = 0  # Sudut target (tegak lurus)
    
    # Menambahkan gangguan awal pada sudut (disturbance)
    theta[0] = disturbance  # Gangguan awal pada sudut
    
    for i in range(1, len(time)):
        error = desired_theta - theta[i - 1]
        integral_error += error * dt
        torque = Kp * error + Ki * integral_error

        # Percepatan sudut (theta_ddot)
        theta_ddot = (torque - mass_pendulum * gravity * length_pendulum * np.sin(theta[i - 1]) 
                      - damping * theta_dot[i - 1]) / inertia
        
        # Memperbarui kecepatan sudut dan sudut
        theta_dot[i] = theta_dot[i - 1] + theta_ddot * dt
        theta[i] = theta[i - 1] + theta_dot[i] * dt

    return theta  # Return the angle for animation

# Fungsi biaya untuk optimasi PI (menghitung error posisi pendulum)
def cost_function(Kp, Ki, disturbance=0.0):
    # Simulasi sistem untuk parameter PI yang diberikan
    theta_best = simulate_pendulum(Kp, Ki, disturbance)
    
    # Fungsi biaya: Integral kesalahan (jumlah kuadrat dari error sudut)
    error = np.sum((theta_best - 0)**2)  # Kesalahan dari posisi tegak
    return error

# Algoritma Artificial Bee Colony untuk optimasi PI
def artificial_bee_colony(population_size, max_iter, disturbance=0.0):
    # Inisialisasi populasi
    population = np.random.uniform(0, 100, (population_size, 2))  # Kp dan Ki inisialisasi acak
    fitness = np.zeros(population_size)
    
    # Daftar untuk menyimpan nilai fitness terbaik di setiap iterasi
    best_fitness_over_time = []
    best_solution_over_time = []

    for i in range(population_size):
        fitness[i] = cost_function(population[i, 0], population[i, 1], disturbance)
    
    best_fitness = np.min(fitness)
    best_solution = population[np.argmin(fitness)]
    
    best_fitness_over_time.append(best_fitness)
    best_solution_over_time.append(best_solution)
    
    # Iterasi ABC
    for iteration in range(max_iter):
        for i in range(population_size):
            candidate = population[i] + np.random.uniform(-1, 1, 2)  # Membuat solusi kandidat baru
            
            # Batasi nilai Kp dan Ki agar tetap dalam rentang yang masuk akal
            candidate = np.clip(candidate, 0, 100)
            
            candidate_fitness = cost_function(candidate[0], candidate[1], disturbance)
            
            if candidate_fitness < fitness[i]:
                population[i] = candidate
                fitness[i] = candidate_fitness
                
                # Memperbarui solusi terbaik
                if candidate_fitness < best_fitness:
                    best_fitness = candidate_fitness
                    best_solution = candidate
        
        # Menyimpan hasil terbaik setiap iterasi
        best_fitness_over_time.append(best_fitness)
        best_solution_over_time.append(best_solution)
        
        # Menampilkan hasil iterasi
        print(f"Iteration {iteration+1}: Best Fitness = {best_fitness:.5f}, Best Kp = {best_solution[0]:.5f}, Best Ki = {best_solution[1]:.5f}")
    
    # Menampilkan hasil iterasi dalam bentuk grafik
    plt.figure()
    plt.plot(best_fitness_over_time)
    plt.title('Convergence of ABC Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Best Fitness')
    plt.grid(True)
    plt.show()
    
    return best_solution  # Mengembalikan Kp dan Ki terbaik

# Optimasi PI menggunakan ABC
population_size = 30
max_iter = 100
disturbance = np.random.uniform(-np.pi / 4, np.pi / 4)  # Gangguan acak antara -45 derajat dan 45 derajat
best_Kp, best_Ki = artificial_bee_colony(population_size, max_iter, disturbance)

print(f"Optimal Kp: {best_Kp}, Optimal Ki: {best_Ki}")

# Simulasi dengan parameter PI terbaik dan gangguan awal
theta_best = simulate_pendulum(best_Kp, best_Ki, disturbance)

# Membuat animasi
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-1, 1)
ax.set_ylim(-0.5, 1)
ax.set_title("Inverted Pendulum Animation with ABC Optimized PI")
ax.set_xlabel("Position (m)")
ax.set_ylabel("Height (m)")
ax.grid()

# Elemen animasi
line, = ax.plot([], [], 'o-', lw=2)
wheel_radius = 0.1  # Radius roda (lingkaran)
wheel = plt.Circle((0, 0), wheel_radius, color='blue', fill=True)  # Membuat roda sebagai lingkaran

ax.add_artist(wheel)  # Menambahkan lingkaran ke plot

# Inisialisasi animasi
def init():
    line.set_data([], [])
    wheel.set_center((0, 0))  # Set posisi awal roda
    return line, wheel

# Update animasi
def update(frame):
    # Posisi pendulum (x dan y)
    x_pendulum = length_pendulum * np.sin(theta_best[frame])
    y_pendulum = length_pendulum * np.cos(theta_best[frame])

    # Pergerakan roda mengikuti posisi pendulum (hubungan horizontal)
    x_wheel = -x_pendulum  # Posisi roda bergerak sesuai dengan pergerakan pendulum
    
    # Posisi pendulum
    line.set_data([x_wheel, x_pendulum], [0, y_pendulum])
    
    # Update posisi roda (lingkaran)
    wheel.set_center((x_wheel, 0))  # Posisi roda mengikuti sumbu horizontal pendulum
    
    return line, wheel

# Membuat animasi
ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=dt * 1000)
plt.show()

