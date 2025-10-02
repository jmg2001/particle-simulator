import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from dataclasses import dataclass
from typing import List
import random


def position_to_grid(position, cell_size):
    grid_x = int(position[0] / cell_size)
    grid_y = int(position[1] / cell_size)
    grid_z = int(position[2] / cell_size)
    return (grid_x, grid_y, grid_z)


def get_neighbor_cells(cell_index):
    cx, cy, cz = cell_index
    neighbors = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                neighbors.append((cx + dx, cy + dy, cz + dz))

    return neighbors


def check_collision(particle_a, particle_b):
    # Calcular distancia
    delta = particle_a.position - particle_b.position
    distance = np.linalg.norm(delta)

    # Radio combinado (asumiendo todas las partículas del mismo tamaño)
    min_distance = particle_a.radius + particle_b.radius

    if distance < min_distance and distance > 0.0001:
        # HAY COLISIÓN

        # 1. Separar partículas
        direction = delta / distance  # Normalizar
        overlap = min_distance - distance

        # Mover cada partícula la mitad del overlap
        particle_a.position += direction * (overlap * 0.5)
        particle_b.position -= direction * (overlap * 0.5)

        # 2. Resolver velocidades (colisión elástica simplificada)
        relative_velocity = particle_a.velocity - particle_b.velocity
        velocity_along_normal = np.dot(relative_velocity, direction)

        # Solo resolver si se están acercando
        if velocity_along_normal < 0:
            # Coeficiente de restitución (bounce)
            restitution = 0.8

            # Impulso
            impulse = -(1 + restitution) * velocity_along_normal
            impulse /= 2  # Dividir entre ambas partículas (masa igual)

            # Aplicar impulso
            particle_a.velocity += direction * impulse
            particle_b.velocity -= direction * impulse


class Particle:
    """Estructura de una partícula"""

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        color: np.ndarray,
        life: float,
        size: float,
    ):
        self.position = position
        self.velocity = velocity
        self.color = color
        self.life = life
        self.size = size
        self.gravity = np.array([0.0, -200, 0.0], dtype=np.float32)

    def update(self, delta_time):
        # Aplicar gravedad
        self.velocity += self.gravity * delta_time

        # Actualizar posición
        self.position += self.velocity * delta_time

        # Actualizar vida
        self.life -= delta_time * 0.5
        self.color[3] = self.life


class ParticleSystem:
    """Sistema de gestión de partículas"""

    def __init__(self, max_particles: int = 10000):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        # self.gravity = np.array([0.0, -1, 0.0], dtype=np.float32)

        # Configuración de colisiones
        self.cell_size = 100  # Tamaño de celda del grid
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
                size=int(2 * (self.cell_size / 3)),
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

            # 1. SEPARAR PARTÍCULAS
            # Calcular cuánto se superponen
            overlap = min_distance - distance

            # Mover cada partícula la mitad del overlap
            # (asumiendo masas iguales)
            separation = direction * (overlap * 0.5)
            particle_a.position += separation
            particle_b.position -= separation

            # 2. RESOLVER VELOCIDADES (Colisión elástica)
            # Velocidad relativa
            relative_velocity = particle_a.velocity - particle_b.velocity

            # Velocidad a lo largo de la normal (dirección de colisión)
            velocity_along_normal = np.dot(relative_velocity, direction)

            # Solo resolver si las partículas se están acercando
            # (evita resolver colisiones múltiples)
            if velocity_along_normal < 0:
                # Calcular impulso
                # Fórmula: j = -(1 + e) * v_rel · n / (1/m_a + 1/m_b)
                # Con masas iguales: j = -(1 + e) * v_rel · n / 2
                impulse_magnitude = -(1.0 + self.restitution) * velocity_along_normal
                impulse_magnitude /= 2.0  # Dividir entre ambas partículas

                # Vector de impulso
                impulse = direction * impulse_magnitude

                # Aplicar impulso a las velocidades
                particle_a.velocity += impulse
                particle_b.velocity -= impulse
                # AL FINAL, después de aplicar impulso:
                MAX_VELOCITY = 15.0  # Ajustar según necesites

                # Limitar velocidad de particle_a
                speed_a = np.linalg.norm(particle_a.velocity)
                if speed_a > MAX_VELOCITY:
                    particle_a.velocity = (particle_a.velocity / speed_a) * MAX_VELOCITY

                # Limitar velocidad de particle_b
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


class ParticleSimulator:
    """Aplicación principal del simulador"""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.window = None
        self.particle_system = None
        self.last_time = 0.0
        self.last_point = 0.0
        self.clicked = False

    def init(self):
        """Inicializar GLFW y OpenGL"""
        if not glfw.init():
            raise Exception("No se pudo inicializar GLFW")

        # Configurar ventana
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self.window = glfw.create_window(
            self.width, self.height, "Simulador de Partículas - Python", None, None
        )

        if not self.window:
            glfw.terminate()
            raise Exception("No se pudo crear la ventana")

        glfw.make_context_current(self.window)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_size_callback)

        # Configurar OpenGL
        glViewport(0, 0, self.width, self.height)
        glClearColor(0.1, 0.1, 0.15, 1.0)

        # Crear sistema de partículas
        self.particle_system = ParticleSystem(max_particles=1000)
        self.particle_system.init_gl()

        self.last_time = glfw.get_time()

    def framebuffer_size_callback(self, window, width, height):
        """Callback para redimensionar ventana"""
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)

    def process_input(self):
        """Procesar entrada del usuario"""
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

        # Emitir partículas con clic del mouse
        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            (x_pixel, y_pixel) = glfw.get_cursor_pos(self.window)
            y_pixel = self.height - y_pixel
            self.particle_system.emit(
                np.array([x_pixel, y_pixel, 0.0], dtype=np.float32), 1
            )
        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            print("hola")
            self.particle_system.reset()

    def run(self):
        """Loop principal"""
        while not glfw.window_should_close(self.window):
            # Calcular delta time
            current_time = glfw.get_time()
            delta_time = current_time - self.last_time
            self.last_time = current_time

            # Procesar input
            self.process_input()

            if current_time - self.last_point > 1:
                # Emitir partículas continuamente
                self.particle_system.emit(
                    np.array([self.width / 2, self.height / 2, 0.0], dtype=np.float32),
                    1,
                )
                self.last_point = current_time

            # Actualizar partículas
            self.particle_system.update(delta_time, self.width, self.height)

            # Renderizar
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Define los límites del mundo 2D (puedes usar el tamaño de la ventana)
            left = 0.0
            right = self.width
            bottom = 0.0
            top = self.height
            near = -1.0
            far = 1.0

            # Matriz de proyección ortográfica
            projection = pyrr.matrix44.create_orthogonal_projection_matrix(
                left, right, bottom, top, near, far
            )
            view = pyrr.matrix44.create_identity()

            # # Matrices de proyección y vista
            # projection = pyrr.matrix44.create_perspective_projection_matrix(
            #     45.0, self.width / self.height, 0.1, 100.0
            # )

            # view = pyrr.matrix44.create_look_at(
            #     np.array([0.0, 3.0, 15.0]),
            #     np.array([0.0, 0.0, 0.0]),
            #     np.array([0.0, 1.0, 0.0]),
            # )

            self.particle_system.render(projection, view)

            # Swap buffers y poll eventos
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        # Limpieza
        self.particle_system.cleanup()
        glfw.terminate()


def main():
    """Función principal"""
    try:
        simulator = ParticleSimulator()
        simulator.init()
        simulator.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
