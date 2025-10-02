import numpy as np
import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
from dataclasses import dataclass
from typing import List
import random


@dataclass
class Particle:
    """Estructura de una partícula"""

    position: np.ndarray
    velocity: np.ndarray
    color: np.ndarray
    life: float
    size: float


class ParticleSystem:
    """Sistema de gestión de partículas"""

    def __init__(self, max_particles: int = 10000):
        self.max_particles = max_particles
        self.particles: List[Particle] = []
        self.gravity = np.array([0.0, -9.8, 0.0], dtype=np.float32)

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

            # Velocidad aleatoria
            velocity = np.array(
                [
                    random.uniform(-2.0, 2.0),
                    random.uniform(-2.0, 2.0),
                    random.uniform(-2.0, 2.0),
                ],
                dtype=np.float32,
            )

            # Color aleatorio
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
                life=1.0,
                size=10.0,
            )

            self.particles.append(particle)

    def update(self, delta_time: float):
        """Actualizar todas las partículas"""
        particles_to_remove = []

        for i, particle in enumerate(self.particles):
            # Aplicar gravedad
            particle.velocity += self.gravity * delta_time

            # Actualizar posición
            particle.position += particle.velocity * delta_time

            # Actualizar vida
            particle.life -= delta_time * 0.5
            particle.color[3] = particle.life  # Alpha
            particle.size = 10.0 * particle.life

            # Marcar para eliminación si murió
            if particle.life <= 0.0:
                particles_to_remove.append(i)

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


class ParticleSimulator:
    """Aplicación principal del simulador"""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.window = None
        self.particle_system = None
        self.last_time = 0.0

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
        self.particle_system = ParticleSystem(max_particles=10000)
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
            self.particle_system.emit(np.array([0.0, 2.0, 0.0], dtype=np.float32), 100)

    def run(self):
        """Loop principal"""
        while not glfw.window_should_close(self.window):
            # Calcular delta time
            current_time = glfw.get_time()
            delta_time = current_time - self.last_time
            self.last_time = current_time

            # Procesar input
            self.process_input()

            # Emitir partículas continuamente
            self.particle_system.emit(np.array([0.0, 2.0, 0.0], dtype=np.float32), 50)

            # Actualizar partículas
            self.particle_system.update(delta_time)

            # Renderizar
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Matrices de proyección y vista
            projection = pyrr.matrix44.create_perspective_projection_matrix(
                45.0, self.width / self.height, 0.1, 100.0
            )

            view = pyrr.matrix44.create_look_at(
                np.array([0.0, 3.0, 15.0]),
                np.array([0.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
            )

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
