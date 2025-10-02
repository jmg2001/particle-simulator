from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


from typing import List

import random

from particle import *


G = -200


class ParticleSystem:
    """Sistema de gestión de partículas"""

    def __init__(self, max_particles: int = 10000):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        # self.gravity = np.array([0.0, -1, 0.0], dtype=np.float32)

        # Configuración de colisiones
        self.particle_size = 20  # px
        self.cell_size = self.particle_size * 1.5  # Tamaño de celda del grid
        self.restitution = 0.5  # Coeficiente de rebote entre partículas
        self.enable_collisions = True  # Activar/desactivar colisiones

        # Shaders
        self.vertex_shader = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec4 aColor;
        layout (location = 2) in float aSize;

        uniform mat4 projection;
        uniform mat4 view;

        out vec4 vertexColor;

        void main() {
            gl_Position = projection * view * vec4(aPos, 1.0);
            gl_PointSize = aSize;
            vertexColor = aColor;
        }
        """

        self.fragment_shader = """
        #version 330 core
        in vec4 vertexColor;
        out vec4 FragColor;
        
        void main() {
            vec2 coord = gl_PointCoord - vec2(0.5);
            if(length(coord) > 0.5)
                discard;
            FragColor = vertexColor;
        }
        """

        self.shader_program = None
        self.vao = None
        self.vbo = None

    def init_gl(self):
        """Inicializar recursos de OpenGL"""
        # Compilar shaders
        self.shader_program = compileProgram(
            compileShader(self.vertex_shader, GL_VERTEX_SHADER),
            compileShader(self.fragment_shader, GL_FRAGMENT_SHADER),
        )

        # Crear VAO y VBO
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)

        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Configurar atributos
        # Posición (3 floats)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Color (4 floats)
        glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)

        # Tamaño (1 float)
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(28))
        glEnableVertexAttribArray(2)

        glBindVertexArray(0)

    def emit(self, origin: np.ndarray, count: int):
        """Emitir nuevas partículas"""
        for _ in range(count):
            if len(self.particles) >= self.max_particles:
                break

            # Velocity 0
            velocity = np.array(
                [random.uniform(-2, 2), random.uniform(-2, 2), 0], dtype=np.float32
            )

            # Random Color
            color = np.array(
                [
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0),
                    random.uniform(0.5, 1.0),
                    1.0,
                ],
                dtype=np.float32,
            )

            particle = Particle(
                position=origin.copy(),
                velocity=velocity,
                color=color,
                life=10.0,
                size=self.particle_size,
                gravity=G,
            )

            self.particles.append(particle)

    def reset(self):
        self.particles = []

    def update(self, delta_time: float, width, height):
        """Actualizar todas las partículas"""
        particles_to_remove = []

        # 1. Aplicar fuerzas y actualizar posiciones
        for i, particle in enumerate(self.particles):
            particle.update(delta_time)

            # Marcar para eliminación si murió
            if particle.life <= 0.0:
                particles_to_remove.append(i)

        self.resolver_collisions_with_world(width, height)

        # 2. Resolver colisiones entre partículas usando spatial grid
        if len(self.particles) > 1:
            grid = self.build_spatial_grid()
            self.resolve_collisions_with_grid(grid)

        # Eliminar partículas muertas (en orden inverso)
        for i in reversed(particles_to_remove):
            self.particles.pop(i)

    def render(self, projection: np.ndarray, view: np.ndarray):
        """Renderizar todas las partículas"""
        if not self.particles:
            return

        # Preparar datos para el buffer
        particle_data = []

        for p in self.particles:
            particle_data.extend(p.position)
            particle_data.extend(p.color)
            particle_data.append(p.size)

        particle_array = np.array(particle_data, dtype=np.float32)

        # Usar shader
        glUseProgram(self.shader_program)

        # Establecer matrices
        proj_loc = glGetUniformLocation(self.shader_program, "projection")
        view_loc = glGetUniformLocation(self.shader_program, "view")
        glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
        glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

        # Actualizar VBO
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(
            GL_ARRAY_BUFFER, particle_array.nbytes, particle_array, GL_DYNAMIC_DRAW
        )

        # Configurar blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_PROGRAM_POINT_SIZE)

        # Dibujar
        glDrawArrays(GL_POINTS, 0, len(self.particles))

        glBindVertexArray(0)

    def cleanup(self):
        """Liberar recursos"""
        if self.vao:
            glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            glDeleteBuffers(1, [self.vbo])
        if self.shader_program:
            glDeleteProgram(self.shader_program)

    def _get_neighbor_cells(self, cell_index: tuple) -> list:
        """
        Obtener las 27 celdas vecinas (3x3x3) incluyendo la celda actual.
        """
        cx, cy, cz = cell_index
        neighbors = []
        dz = 0

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                # for dz in [-1, 0, 1]:
                neighbors.append((cx + dx, cy + dy, cz + dz))

        return neighbors

    def _check_and_resolve_collision(self, particle_a: Particle, particle_b: Particle):
        """
        Detectar y resolver colisión entre dos partículas.
        Implementa separación de posiciones y respuesta de velocidad.
        """
        # Calcular vector entre partículas
        delta = particle_a.position - particle_b.position
        distance = np.linalg.norm(delta)

        # Radio combinado
        min_distance = particle_a.size / 2 + particle_b.size / 2

        # Chequear si hay colisión
        if distance < min_distance and distance > 0.0001:
            # Normalizar dirección
            direction = delta / distance

            # 1. SEPARAR PARTÍCULAS CON LÍMITE
            overlap = min_distance - distance

            # Limitar separación máxima por frame (evita explosiones)
            MAX_SEPARATION_PER_FRAME = 0.2
            overlap = min(overlap, MAX_SEPARATION_PER_FRAME)

            # Separar con un pequeño extra para evitar que se queden pegadas
            SEPARATION_BIAS = 0.01  # Pequeño empujón extra
            effective_overlap = overlap + SEPARATION_BIAS

            # Mover cada partícula la mitad del overlap
            separation = direction * (effective_overlap * 0.5)
            particle_a.position += separation
            particle_b.position -= separation

            # 2. RESOLVER VELOCIDADES (Colisión elástica mejorada)
            relative_velocity = particle_a.velocity - particle_b.velocity
            velocity_along_normal = np.dot(relative_velocity, direction)

            # Solo resolver si las partículas se están acercando
            # (evita resolver colisiones múltiples)
            if velocity_along_normal < 0:
                # Ajustar restitución según penetración
                penetration_ratio = overlap / min_distance

                # Si están muy superpuestas, reducir rebote para estabilidad
                if penetration_ratio > 0.5:
                    effective_restitution = self.restitution * 0.4
                else:
                    effective_restitution = self.restitution

                # Calcular impulso
                # Fórmula: j = -(1 + e) * v_rel · n / (1/m_a + 1/m_b)
                # Con masas iguales: j = -(1 + e) * v_rel · n / 2
                impulse_magnitude = (
                    -(1.0 + effective_restitution) * velocity_along_normal
                )
                impulse_magnitude /= 2.0  # Dividir entre ambas partículas

                # Vector de impulso
                impulse = direction * impulse_magnitude

                # Aplicar impulso a las velocidades
                particle_a.velocity += impulse
                particle_b.velocity -= impulse

                # 3. LIMITAR VELOCIDADES (evita que salgan disparadas)
                MAX_VELOCITY = 15.0

                speed_a = np.linalg.norm(particle_a.velocity)
                if speed_a > MAX_VELOCITY:
                    particle_a.velocity = (particle_a.velocity / speed_a) * MAX_VELOCITY

                speed_b = np.linalg.norm(particle_b.velocity)
                if speed_b > MAX_VELOCITY:
                    particle_b.velocity = (particle_b.velocity / speed_b) * MAX_VELOCITY

    def build_spatial_grid(self) -> dict:
        """
        Construir un grid espacial para optimizar detección de colisiones.
        Retorna un diccionario donde la clave es (grid_x, grid_y, grid_z)
        y el valor es una lista de índices de partículas en esa celda.
        """
        grid = {}

        for idx, particle in enumerate(self.particles):
            # Calcular índice de celda para esta partícula
            cell_x = int(particle.position[0] / self.cell_size)
            cell_y = int(particle.position[1] / self.cell_size)
            cell_z = int(particle.position[2] / self.cell_size)

            cell_index = (cell_x, cell_y, cell_z)

            # Agregar partícula a la celda
            if cell_index not in grid:
                grid[cell_index] = []

            grid[cell_index].append(idx)

        return grid

    def resolver_collisions_with_world(self, width, height):
        for particle in self.particles:
            # Suelo
            if particle.position[1] - (particle.size / 2) < 0:
                particle.position[1] = particle.size / 2
                particle.velocity[1] = (
                    -particle.velocity[1] * 0.6
                )  # rebote con pérdida de energía

            # Paredes
            if particle.position[0] - (particle.size / 2) < 0:
                particle.position[0] = particle.size / 2
                particle.velocity[0] = (
                    -particle.velocity[0] * 0.6
                )  # rebote con pérdida de energía

            if particle.position[0] + (particle.size / 2) > width:
                particle.position[0] = width - particle.size / 2
                particle.velocity[0] = (
                    -particle.velocity[0] * 0.6
                )  # rebote con pérdida de energía

    def resolve_collisions_with_grid(self, grid: dict):
        """
        Resolver colisiones entre partículas usando el spatial grid.
        Solo chequea colisiones entre partículas en celdas vecinas.
        """
        # Conjunto para evitar chequear el mismo par dos veces
        checked_pairs = set()

        # Para cada celda que contiene partículas
        for cell_index, particle_indices in grid.items():
            # Obtener celdas vecinas (27 celdas: la actual + 26 vecinas)
            neighbor_cells = self._get_neighbor_cells(cell_index)

            # Recolectar índices de todas las partículas vecinas
            nearby_indices = []
            for neighbor_index in neighbor_cells:
                if neighbor_index in grid:
                    nearby_indices.extend(grid[neighbor_index])

            # Chequear colisiones entre partículas de esta celda y vecinas
            for idx_a in particle_indices:
                particle_a = self.particles[idx_a]

                for idx_b in nearby_indices:
                    # No colisionar consigo misma
                    if idx_a == idx_b:
                        continue

                    # Evitar chequear el mismo par dos veces
                    pair = (min(idx_a, idx_b), max(idx_a, idx_b))
                    if pair in checked_pairs:
                        continue

                    checked_pairs.add(pair)

                    # Chequear y resolver colisión
                    particle_b = self.particles[idx_b]
                    self._check_and_resolve_collision(particle_a, particle_b)
