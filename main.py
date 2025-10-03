from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader

import glfw

import pyrr
import math

from particle import *
from particleSystem import *


class ParticleSimulator:
    """Aplicación principal del simulador"""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.window = None
        self.particle_system = None
        self.last_time = 0.0
        self.last_point = 0.0
        self.emmit_particle_periodly = False

        self.mouse_particle = None
        self.particle_emmited = False

        self.last_mouse_pos = None
        self.last_time = glfw.get_time()
        self.mouse_velocity = [0.0, 0.0]
        self.alpha = 0.2  # factor de suavizado

    def update_mouse_velocity(self, current_pos):
        current_time = glfw.get_time()
        dt = current_time - self.last_time
        self.last_time = current_time

        if self.last_mouse_pos is not None and dt > 0:
            dx = current_pos[0] - self.last_mouse_pos[0]
            dy = current_pos[1] - self.last_mouse_pos[1]
            vx = dx / dt
            vy = dy / dt

            k = 0.8

            self.mouse_velocity[0] = (
                self.alpha * vx + (1 - self.alpha) * self.mouse_velocity[0]
            ) * k
            self.mouse_velocity[1] = (
                self.alpha * vy + (1 - self.alpha) * self.mouse_velocity[1]
            ) * k

        self.last_mouse_pos = current_pos

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

    def clamp_velocity(self, vmin=-1, vmax=1):
        MAX_VEL = 100
        MIN_VEL = -100

        if self.mouse_velocity[0] < 0:
            if self.mouse_velocity[0] < MIN_VEL:
                self.mouse_velocity[0] = MIN_VEL
        else:
            if self.mouse_velocity[0] > MAX_VEL:
                self.mouse_velocity[0] = MAX_VEL

        if self.mouse_velocity[1] < 0:
            if self.mouse_velocity[1] < MIN_VEL:
                self.mouse_velocity[1] = MIN_VEL
        else:
            if self.mouse_velocity[1] > MAX_VEL:
                self.mouse_velocity[1] = MAX_VEL

    def process_input(self):
        """Procesar entrada del usuario"""
        if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(self.window, True)

        # Emitir partículas con clic del mouse
        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:

            (x_pixel, y_pixel) = glfw.get_cursor_pos(self.window)
            y_pixel = self.height - y_pixel

            self.update_mouse_velocity([x_pixel, y_pixel])

            self.particle_emmited = False

            if self.mouse_particle == None:
                # Velocity
                velocity = np.array([0, 0, 0], dtype=np.float32)
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
                    position=np.array([x_pixel, y_pixel, 0.0], dtype=np.float32),
                    velocity=velocity,
                    color=color,
                    life=10.0,
                    size=15.0,
                    gravity=G,
                )

                self.mouse_particle = particle

            else:
                self.mouse_particle.position = np.array(
                    [x_pixel, y_pixel, 0.0], dtype=np.float32
                )

        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE:
            if self.mouse_particle != None:
                self.clamp_velocity()
                self.mouse_particle.velocity[0] = self.mouse_velocity[0]
                self.mouse_particle.velocity[1] = self.mouse_velocity[1]
                self.particle_system.emit_mouse(self.mouse_particle)
                self.mouse_particle = None
                self.particle_emmited = True

        if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS:
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

            if self.emmit_particle_periodly:
                # Every second
                if current_time - self.last_point > 1:
                    # Emitir partículas continuamente
                    self.particle_system.emit(
                        np.array(
                            [random.randint(0, self.width), self.height / 2, 0.0],
                            dtype=np.float32,
                        ),
                        1,
                    )
                    self.last_point = current_time

            # Actualizar partículas
            if self.particle_emmited != None:
                self.particle_system.update(
                    delta_time, self.width, self.height, self.mouse_particle
                )
            else:
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
