import numpy as np
from sklearn import neighbors
from tqdm import tqdm

MAX_PARTICLES = 125
DOMAIN_WIDTH = 40
DOMAIN_HEIGHT = 80

PARTICLE_MASS = 1
ISOTROPIC_EXPONENT = 20
BASE_DENSITY = 1
SMOOTHING_LENGTH = 5
DYNAMIC_VISCOSITY = 0.5
DAMPING_COEFFICIENT = -0.9
CONSTANT_FORCE = np.array([[0.0, -1.5]])

TIME_STEP_LENGTH = 0.01
N_TIME_STEPS = 10000   # reduzido para não gerar dados gigantes
ADD_PARTICLES_EVERY = 25

DOMAIN_X_LIM = np.array([
    SMOOTHING_LENGTH,
    DOMAIN_WIDTH - SMOOTHING_LENGTH,
])
DOMAIN_Y_LIM = np.array([
    SMOOTHING_LENGTH,
    DOMAIN_HEIGHT - SMOOTHING_LENGTH,
])

NORMALIZATION_DENSITY = (
    (315 * PARTICLE_MASS) / (64 * np.pi * SMOOTHING_LENGTH**9)
)
NORMALIZATION_PRESSURE_FORCE = (
    -(45 * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)
)
NORMALIZATION_VISCOUS_FORCE = (
    (45 * DYNAMIC_VISCOSITY * PARTICLE_MASS) / (np.pi * SMOOTHING_LENGTH**6)
)


def run_sph_simulation():
    """Roda a simulação SPH e retorna as posições das partículas em cada passo de tempo.

    Retorna:
        positions_over_time (list of np.ndarray): lista de arrays (n_particles, 2)
    """
    n_particles = 0
    positions = np.zeros((n_particles, 2))
    velocities = np.zeros_like(positions)

    positions_over_time = []

    for iter in tqdm(range(N_TIME_STEPS)):
        # Adiciona novas partículas no topo
        if iter % ADD_PARTICLES_EVERY == 0 and n_particles < MAX_PARTICLES:
            new_positions = np.array([
                [10 + np.random.rand(), DOMAIN_Y_LIM[1]],
            ])
            new_velocities = np.array([
                [-3.0, -15.0],
            ])

            n_particles += new_positions.shape[0]
            positions = np.concatenate((positions, new_positions), axis=0)
            velocities = np.concatenate((velocities, new_velocities), axis=0)

        # Se não há partículas, pula o passo
        if n_particles == 0:
            positions_over_time.append(positions.copy())
            continue

        # Busca de vizinhos
        if n_particles > 1:
            neighbor_ids, distances = neighbors.KDTree(
                positions,
            ).query_radius(
                positions,
                SMOOTHING_LENGTH,
                return_distance=True,
                sort_results=True,
            )

            densities = np.zeros(n_particles)
            for i in range(n_particles):
                for j_in_list, j in enumerate(neighbor_ids[i]):
                    d = distances[i][j_in_list]
                    if d < SMOOTHING_LENGTH:  # segurança
                        densities[i] += NORMALIZATION_DENSITY * (
                            SMOOTHING_LENGTH**2 - d**2
                        )**3

            # Evita densidade zero
            densities = np.maximum(densities, 1e-8)

            pressures = ISOTROPIC_EXPONENT * (densities - BASE_DENSITY)
            forces = np.zeros_like(positions)

            # Remove o próprio ponto
            neighbor_ids = [np.delete(x, 0) for x in neighbor_ids]
            distances = [np.delete(x, 0) for x in distances]

            for i in range(n_particles):
                for j_in_list, j in enumerate(neighbor_ids[i]):
                    d = distances[i][j_in_list]
                    if d < 1e-8:  # evita divisão por zero
                        continue

                    # Força de pressão
                    forces[i] += NORMALIZATION_PRESSURE_FORCE * (
                        -(positions[j] - positions[i]) / d
                        * (pressures[j] + pressures[i]) / (2 * densities[j])
                        * (SMOOTHING_LENGTH - d)**2
                    )

                    # Força viscosa
                    forces[i] += NORMALIZATION_VISCOUS_FORCE * (
                        (velocities[j] - velocities[i]) / densities[j]
                        * (SMOOTHING_LENGTH - d)
                    )
        else:
            # Só 1 partícula: densidade básica + apenas força externa
            densities = np.full(n_particles, BASE_DENSITY)
            forces = np.zeros_like(positions)

        # Gravidade
        forces += CONSTANT_FORCE

        # Euler Step
        velocities = velocities + TIME_STEP_LENGTH * forces / densities[:, np.newaxis]
        positions = positions + TIME_STEP_LENGTH * velocities

        # Condições de contorno
        out_of_left_boundary = positions[:, 0] < DOMAIN_X_LIM[0]
        out_of_right_boundary = positions[:, 0] > DOMAIN_X_LIM[1]
        out_of_bottom_boundary = positions[:, 1] < DOMAIN_Y_LIM[0]
        out_of_top_boundary = positions[:, 1] > DOMAIN_Y_LIM[1]

        velocities[out_of_left_boundary, 0] *= DAMPING_COEFFICIENT
        positions[out_of_left_boundary, 0] = DOMAIN_X_LIM[0]

        velocities[out_of_right_boundary, 0] *= DAMPING_COEFFICIENT
        positions[out_of_right_boundary, 0] = DOMAIN_X_LIM[1]

        velocities[out_of_bottom_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_bottom_boundary, 1] = DOMAIN_Y_LIM[0]

        velocities[out_of_top_boundary, 1] *= DAMPING_COEFFICIENT
        positions[out_of_top_boundary, 1] = DOMAIN_Y_LIM[1]

        # Verificação final contra NaN
        if np.isnan(positions).any() or np.isnan(velocities).any():
            print(f"⚠️ NaN detectado na iteração {iter}, abortando simulação.")
            break

        # Guarda as posições atuais (cópia)
        positions_over_time.append(positions.copy())

    return positions_over_time
