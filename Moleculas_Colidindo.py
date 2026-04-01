from manim import *
import numpy as np
import random

# --- Funções de Geração de Moléculas (com pequena correção no raio de H2O) ---
def create_h2o_molecule(molecule_radius=0.05):
    """Cria uma molécula de H2O. O raio de colisão é definido pelo átomo de oxigênio."""
    h_radius = molecule_radius * 0.8
    o_radius = molecule_radius * 1.2
    oxigenio = Dot(color=BLUE, radius=o_radius)
    hidrogenio1 = Dot(color=RED, radius=h_radius).shift(LEFT * o_radius * 0.6 + DOWN * o_radius * 0.6)
    hidrogenio2 = Dot(color=RED, radius=h_radius).shift(RIGHT * o_radius * 0.6 + DOWN * o_radius * 0.6)
    molecule = VGroup(oxigenio, hidrogenio1, hidrogenio2)
    molecule.velocidade = np.array([0.0, 0.0, 0.0])
    molecule.rotacao_velocidade = np.random.uniform(-PI/2, PI/2)
    molecule.radius = o_radius  # Raio de colisão baseado no átomo maior
    return molecule

def create_o2_molecule(molecule_radius=0.25):
    """Cria uma molécula de O2."""
    oxigenio1 = Dot(color=BLUE, radius=molecule_radius)
    oxigenio2 = Dot(color=BLUE, radius=molecule_radius).shift(RIGHT * molecule_radius * 2)
    molecule = VGroup(oxigenio1, oxigenio2)
    molecule.velocidade = np.array([0.0, 0.0, 0.0])
    molecule.rotacao_velocidade = np.random.uniform(-PI/2, PI/2)
    molecule.radius = molecule_radius
    return molecule

def run_simulation(scene, num_moleculas, velocidade_inicial_max, tamanho_moleculas, molecule_factory): 
    # --- paredes separadas ---
    parede_left   = Line([-2, -3, 0], [-2,  1, 0], color=WHITE).set_z_index(1)
    parede_right  = Line([ 2, -3, 0], [ 2,  1, 0], color=WHITE).set_z_index(1)
    parede_bottom = Line([-2, -3, 0], [ 2, -3, 0], color=WHITE).set_z_index(1)
    parede_top    = Line([-2,  1, 0], [ 2,  1, 0], color=WHITE).set_z_index(1)

    recipiente = VGroup(parede_left, parede_right, parede_bottom, parede_top)
    recipiente.ativo = True       # colisão com paredes
    recipiente.congelado = False  # pausa geral da física
    scene.add(recipiente)

    x_min, x_max = -2, 2
    y_min, y_max = -3, 1

    moleculas = VGroup()
    for _ in range(num_moleculas):
        m = molecule_factory(tamanho_moleculas)
        x = np.random.uniform(x_min + m.radius, x_max - m.radius)
        y = np.random.uniform(y_min + m.radius, y_max - m.radius)
        m.move_to([x, y, 0])
        m.velocidade = np.array([
            np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max),
            np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max),
            0
        ])
        moleculas.add(m)
    scene.add(moleculas)

    # --- updater que controla movimento e colisões ---
    def update_moleculas(mob, dt):
        # Se estiver congelado, não faz nada
        if recipiente.congelado:
            return

        for mol in mob:
            mol.shift(mol.velocidade * dt)
            mol.rotate(mol.rotacao_velocidade * dt)

            # colisão só acontece se recipiente.ativo == True
            if recipiente.ativo:
                if parede_left in recipiente and mol.get_x() <= x_min + mol.radius:
                    mol.velocidade[0] *= -1
                if parede_right in recipiente and mol.get_x() >= x_max - mol.radius:
                    mol.velocidade[0] *= -1
                if parede_bottom in recipiente and mol.get_y() <= y_min + mol.radius:
                    mol.velocidade[1] *= -1
                if parede_top in recipiente and mol.get_y() >= y_max - mol.radius:
                    mol.velocidade[1] *= -1

        # --- colisão entre partículas ---
        for i in range(len(mob)):
            for j in range(i + 1, len(mob)):
                mol1 = mob[i]
                mol2 = mob[j]
                
                distancia = np.linalg.norm(mol1.get_center() - mol2.get_center())
                soma_raios = mol1.radius + mol2.radius

                if distancia < soma_raios:
                    vetor_colisao = mol1.get_center() - mol2.get_center()
                    direcao = vetor_colisao / distancia
                    sobreposicao = soma_raios - distancia
                    mol1.shift(direcao * sobreposicao / 2)
                    mol2.shift(-direcao * sobreposicao / 2)

                    distancia_vetor = mol1.get_center() - mol2.get_center()
                    norma_distancia_vetor = np.linalg.norm(distancia_vetor)
                    if norma_distancia_vetor != 0:
                        v1_prime = mol1.velocidade - np.dot(mol1.velocidade - mol2.velocidade, distancia_vetor) / norma_distancia_vetor**2 * distancia_vetor
                        v2_prime = mol2.velocidade - np.dot(mol2.velocidade - mol1.velocidade, -distancia_vetor) / norma_distancia_vetor**2 * (-distancia_vetor)
                        mol1.velocidade = v1_prime
                        mol2.velocidade = v2_prime

    moleculas.add_updater(update_moleculas)
    return moleculas, recipiente


def run_simulation_circle(
    scene, 
    num_moleculas, 
    velocidade_inicial_max, 
    tamanho_moleculas, 
    molecule_factory, 
    posicoes_iniciais=None,
    velocidades_iniciais=None   # <<< novo parâmetro
):
    raio_recipiente = 2.5
    recipiente = Circle(radius=raio_recipiente, color=WHITE).set_z_index(1)
    recipiente.ativo = True
    recipiente.congelado = False
    scene.play(Create(recipiente))

    moleculas = VGroup()
    for i in range(num_moleculas):
        m = molecule_factory(tamanho_moleculas)

        # --- Posição ---
        if posicoes_iniciais is not None:
            x, y = posicoes_iniciais[i]
        else:
            while True:
                x = np.random.uniform(-raio_recipiente, raio_recipiente)
                y = np.random.uniform(-raio_recipiente, raio_recipiente)
                if x**2 + y**2 <= (raio_recipiente - m.radius)**2:
                    break
        m.move_to([x, y, 0])

        # --- Velocidade ---
        if velocidades_iniciais is not None:
            vx, vy = velocidades_iniciais[i]
        else:
            vx = np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max)
            vy = np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max)

        m.velocidade = np.array([vx, vy, 0])
        moleculas.add(m)

    scene.play(Create(moleculas))

    # update igual ao seu...
    def update_moleculas(mob, dt):
        if recipiente.congelado:
            return  

        if not recipiente.ativo:
            for mol in mob:
                mol.shift(mol.velocidade * dt)
                mol.rotate(mol.rotacao_velocidade * dt)
            return

        # Movimento + colisão borda
        for mol in mob:
            mol.shift(mol.velocidade * dt)
            mol.rotate(mol.rotacao_velocidade * dt)

            centro = mol.get_center()
            dist = np.linalg.norm(centro[:2])
            if dist + mol.radius >= raio_recipiente:
                normal = centro[:2] / dist
                v = mol.velocidade[:2]
                v_refletido = v - 2 * np.dot(v, normal) * normal
                mol.velocidade[:2] = v_refletido
                overlap = dist + mol.radius - raio_recipiente
                mol.shift(-np.append(normal, 0) * overlap)

        # Colisão molécula-molécula
        for i in range(len(mob)):
            for j in range(i + 1, len(mob)):
                mol1, mol2 = mob[i], mob[j]
                distancia = np.linalg.norm(mol1.get_center() - mol2.get_center())
                soma_raios = mol1.radius + mol2.radius
                if distancia < soma_raios:
                    vetor_colisao = mol1.get_center() - mol2.get_center()
                    direcao = vetor_colisao / distancia
                    sobreposicao = soma_raios - distancia
                    mol1.shift(direcao * sobreposicao / 2)
                    mol2.shift(-direcao * sobreposicao / 2)

                    distancia_vetor = mol1.get_center() - mol2.get_center()
                    norma_dist = np.linalg.norm(distancia_vetor)
                    if norma_dist != 0:
                        v1_prime = mol1.velocidade - np.dot(
                            mol1.velocidade - mol2.velocidade, distancia_vetor
                        ) / norma_dist**2 * distancia_vetor
                        v2_prime = mol2.velocidade - np.dot(
                            mol2.velocidade - mol1.velocidade, -distancia_vetor
                        ) / norma_dist**2 * (-distancia_vetor)
                        mol1.velocidade = v1_prime
                        mol2.velocidade = v2_prime

    moleculas.add_updater(update_moleculas)
    return moleculas, recipiente


def run_simulation_circle_gradual_movel(
    scene, 
    num_moleculas, 
    velocidade_inicial_max, 
    tamanho_moleculas, 
    molecule_factory, 
    raio_inicial=2.5,
    posicoes_iniciais=None,
    velocidades_iniciais=None,
    posicao_centro=ORIGIN,  # <<< MUDANÇA: Novo parâmetro de centro
    taxa_aquecimento_por_segundo=1.0
):
    # <<< MUDANÇA: Move o recipiente para a posição_centro >>>
    recipiente = Circle(radius=raio_inicial, color=WHITE).set_z_index(1).move_to(posicao_centro)
    recipiente.ativo = True
    recipiente.congelado = False
    
    recipiente.aquecendo = False 
    recipiente.taxa_aquecimento = taxa_aquecimento_por_segundo
    
    scene.play(Create(recipiente))

    moleculas = VGroup()
    for i in range(num_moleculas):
        m = molecule_factory(tamanho_moleculas)
        if posicoes_iniciais is not None:
            x, y = posicoes_iniciais[i]
        else:
            while True:
                x = np.random.uniform(-raio_inicial, raio_inicial)
                y = np.random.uniform(-raio_inicial, raio_inicial)
                if x**2 + y**2 <= (raio_inicial - m.radius)**2:
                    break
        
        # <<< MUDANÇA: Move a molécula para a posição relativa ao centro >>>
        m.move_to(np.array([x, y, 0]) + posicao_centro)

        if velocidades_iniciais is not None:
            vx, vy = velocidades_iniciais[i]
        else:
            vx = np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max)
            vy = np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max)

        m.velocidade = np.array([vx, vy, 0])
        moleculas.add(m)

    scene.play(Create(moleculas))

    # --- Updater Modificado ---
    def update_moleculas(mob, dt):
        if recipiente.congelado or dt == 0:
            return 
        
        raio_atual = recipiente.get_width() / 2.0
        
        # <<< MUDANÇA: Pega o centro atual do *objeto* recipiente >>>
        centro_atual_recipiente = recipiente.get_center()

        if recipiente.aquecendo:
            fator_dt = recipiente.taxa_aquecimento ** dt
            for mol in mob:
                mol.velocidade *= fator_dt

        if not recipiente.ativo:
            for mol in mob:
                mol.shift(mol.velocidade * dt)
                mol.rotate(mol.rotacao_velocidade * dt)
            return

        # Movimento + colisão borda
        for mol in mob:
            mol.shift(mol.velocidade * dt)
            mol.rotate(mol.rotacao_velocidade * dt)

            centro_mol = mol.get_center()
            
            # <<< MUDANÇA: Física de colisão relativa ao centro_atual_recipiente >>>
            vetor_do_centro = centro_mol[:2] - centro_atual_recipiente[:2]
            dist = np.linalg.norm(vetor_do_centro)
            
            if dist + mol.radius >= raio_atual:
                # <<< MUDANÇA: Normal é baseada no vetor_do_centro >>>
                normal = vetor_do_centro / dist 
                v = mol.velocidade[:2]
                v_refletido = v - 2 * np.dot(v, normal) * normal
                mol.velocidade[:2] = v_refletido
                overlap = dist + mol.radius - raio_atual 
                mol.shift(-np.append(normal, 0) * overlap)

        # Colisão molécula-molécula (sem alteração)
        # (Esta física é relativa entre moléculas, não importa onde está o centro)
        for i in range(len(mob)):
            for j in range(i + 1, len(mob)):
                mol1, mol2 = mob[i], mob[j]
                # ... (resto do código de colisão igual) ...
                distancia = np.linalg.norm(mol1.get_center() - mol2.get_center())
                soma_raios = mol1.radius + mol2.radius
                if distancia < soma_raios:
                    vetor_colisao = mol1.get_center() - mol2.get_center()
                    direcao = vetor_colisao / distancia
                    sobreposicao = soma_raios - distancia
                    mol1.shift(direcao * sobreposicao / 2)
                    mol2.shift(-direcao * sobreposicao / 2)

                    distancia_vetor = mol1.get_center() - mol2.get_center()
                    norma_dist = np.linalg.norm(distancia_vetor)
                    if norma_dist != 0:
                        v1_prime = mol1.velocidade - np.dot(
                            mol1.velocidade - mol2.velocidade, distancia_vetor
                        ) / norma_dist**2 * distancia_vetor
                        v2_prime = mol2.velocidade - np.dot(
                            mol2.velocidade - mol1.velocidade, -distancia_vetor
                        ) / norma_dist**2 * (-distancia_vetor)
                        mol1.velocidade = v1_prime
                        mol2.velocidade = v2_prime

    moleculas.add_updater(update_moleculas)
    return moleculas, recipiente


def create_simulation_mobjects(
    num_moleculas, 
    velocidade_inicial_max, 
    tamanho_moleculas, 
    molecule_factory, 
    raio_inicial=2.5,
    posicoes_iniciais=None,
    velocidades_iniciais=None,
    posicao_centro=ORIGIN,
    taxa_aquecimento_por_segundo=1.0
):
    """
    Cria os Mobjects da simulação (recipiente e moléculas com updater)
    SEM chamar scene.play(). Retorna os Mobjects.
    """
    # Cria o recipiente na posição certa
    recipiente = Circle(radius=raio_inicial, color=WHITE).set_z_index(1).move_to(posicao_centro)
    recipiente.ativo = True
    recipiente.congelado = False
    recipiente.aquecendo = False 
    recipiente.taxa_aquecimento = taxa_aquecimento_por_segundo
    
    # Cria as moléculas na posição certa
    moleculas = VGroup()
    for i in range(num_moleculas):
        m = molecule_factory(tamanho_moleculas)
        if posicoes_iniciais is not None:
            x, y = posicoes_iniciais[i]
        else:
            while True:
                x = np.random.uniform(-raio_inicial, raio_inicial)
                y = np.random.uniform(-raio_inicial, raio_inicial)
                if x**2 + y**2 <= (raio_inicial - m.radius)**2:
                    break
        m.move_to(np.array([x, y, 0]) + posicao_centro) # Adiciona o centro

        if velocidades_iniciais is not None:
            vx, vy = velocidades_iniciais[i]
        else:
            vx = np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max)
            vy = np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max)

        m.velocidade = np.array([vx, vy, 0])
        moleculas.add(m)

    # --- Updater (a lógica da física) ---
    # (Exatamente o mesmo updater da sua função 'run_simulation_circle_gradual_movel')
    def update_moleculas(mob, dt):
        if recipiente.congelado or dt == 0:
            return 
        
        raio_atual = recipiente.get_width() / 2.0
        centro_atual_recipiente = recipiente.get_center()

        if recipiente.aquecendo:
            fator_dt = recipiente.taxa_aquecimento ** dt
            for mol in mob:
                mol.velocidade *= fator_dt

        if not recipiente.ativo:
            for mol in mob:
                mol.shift(mol.velocidade * dt)
                mol.rotate(mol.rotacao_velocidade * dt)
            return

        # Movimento + colisão borda
        for mol in mob:
            mol.shift(mol.velocidade * dt)
            mol.rotate(mol.rotacao_velocidade * dt)
            centro_mol = mol.get_center()
            vetor_do_centro = centro_mol[:2] - centro_atual_recipiente[:2]
            dist = np.linalg.norm(vetor_do_centro)
            
            if dist + mol.radius >= raio_atual:
                normal = vetor_do_centro / dist 
                v = mol.velocidade[:2]
                v_refletido = v - 2 * np.dot(v, normal) * normal
                mol.velocidade[:2] = v_refletido
                overlap = dist + mol.radius - raio_atual 
                mol.shift(-np.append(normal, 0) * overlap)

        # Colisão molécula-molécula
        for i in range(len(mob)):
            for j in range(i + 1, len(mob)):
                mol1, mol2 = mob[i], mob[j]
                distancia = np.linalg.norm(mol1.get_center() - mol2.get_center())
                soma_raios = mol1.radius + mol2.radius
                if distancia < soma_raios:
                    vetor_colisao = mol1.get_center() - mol2.get_center()
                    direcao = vetor_colisao / distancia
                    sobreposicao = soma_raios - distancia
                    mol1.shift(direcao * sobreposicao / 2)
                    mol2.shift(-direcao * sobreposicao / 2)
                    distancia_vetor = mol1.get_center() - mol2.get_center()
                    norma_dist = np.linalg.norm(distancia_vetor)
                    if norma_dist != 0:
                        v1_prime = mol1.velocidade - np.dot(
                            mol1.velocidade - mol2.velocidade, distancia_vetor
                        ) / norma_dist**2 * distancia_vetor
                        v2_prime = mol2.velocidade - np.dot(
                            mol2.velocidade - mol1.velocidade, -distancia_vetor
                        ) / norma_dist**2 * (-distancia_vetor)
                        mol1.velocidade = v1_prime
                        mol2.velocidade = v2_prime

    moleculas.add_updater(update_moleculas)
    
    # Retorna os Mobjects para a cena principal controlar
    return recipiente, moleculas


def gerar_posicoes_aleatorias(num_moleculas, raio_recipiente, tamanho_moleculas):
    posicoes = []
    for _ in range(num_moleculas):
        while True:
            x = np.random.uniform(-raio_recipiente, raio_recipiente)
            y = np.random.uniform(-raio_recipiente, raio_recipiente)
            if x**2 + y**2 <= (raio_recipiente - tamanho_moleculas)**2:
                posicoes.append((x, y))
                break
    return posicoes


def gerar_velocidades_aleatorias(num_moleculas, velocidade_inicial_max):
    velocidades = []
    for _ in range(num_moleculas):
        vx = np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max)
        vy = np.random.uniform(-velocidade_inicial_max, velocidade_inicial_max)
        velocidades.append((vx, vy))
    return velocidades


