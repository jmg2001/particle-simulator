import glfw

import pyrr

from particle import *
from particleSystem import *

from utils import *


class ParticleSimulator:
    """Aplicación principal del simulador"""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self.window = None

        self.view = None
        self.projection = None

        self.particle_system = None
        self.last_time = 0.0

        self.last_time_particle_emitted = 0.0
        self.emit_particle_periodically = False

        self.mouse_particle = None
        self.mosue_particle_emitted = False

        self.last_mouse_pos = None
        self.last_mouse_time = 0.0
        self.mouse_velocity = [0.0, 0.0]
        self.mouse_alpha = 0.9  # factor de suavizado
        self.particles_emitted_with_mouse = 10

        # CONFIGURACIÓN DE TIMESTEP FIJO
        self.fixed_timestep = 1.0 / 60.0  # 60 Hz de física
        self.accumulator = 0.0
        self.max_frame_time = 0.25  # Máximo 250ms por frame

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

        # Define los límites del mundo 2D (puedes usar el tamaño de la ventana)
        LEFT_LIMIT, RIGHT_LIMIT, BOTTOM_LIMIT, TOP_LIMIT = [
            0.0,
            self.width,
            0.0,
            self.height,
        ]
        NEAR = -1.0
        FAR = 1.0

        # Matriz de proyección ortográfica
        self.projection = pyrr.matrix44.create_orthogonal_projection_matrix(
            LEFT_LIMIT, RIGHT_LIMIT, BOTTOM_LIMIT, TOP_LIMIT, NEAR, FAR
        )
        self.view = pyrr.matrix44.create_identity()

    def run(self):
        """Loop principal"""
        while not glfw.window_should_close(self.window):
            # Calcular delta time
            current_time = glfw.get_time()
            frame_time = current_time - self.last_time

            self.last_time = current_time

            # 2. LIMITAR FRAME TIME (evitar saltos enormes)
            if frame_time > 0.25:
                frame_time = 0.25  # Máximo 250ms por frame

            # 3. ACUMULAR TIEMPO
            self.accumulator += frame_time

            # 4. ACTUALIZAR FÍSICA EN PASOS FIJOS
            # Esto garantiza simulación consistente independiente del framerate
            steps = 0
            max_steps = 5  # Evitar "spiral of death"

            while self.accumulator >= self.fixed_timestep and steps < max_steps:

                # Procesar input
                self.process_input()

                if self.emit_particle_periodically:
                    # Every second
                    if current_time - self.last_time_particle_emitted > 1:
                        # Emitir partículas continuamente
                        self.particle_system.emit(
                            particle=Particle(
                                position=get_random_position(self.width, self.height)
                            )
                        )
                        self.last_time_particle_emitted = current_time

                # Actualizar partículas
                self.particle_system.update(
                    self.fixed_timestep, self.width, self.height, self.mouse_particle
                )

                # Reducir acumulador
                self.accumulator -= self.fixed_timestep
                steps += 1

            # Si hubo demasiados pasos, descartar tiempo extra
            if steps >= max_steps:
                self.accumulator = 0.0

            # Renderizar
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # # Matrices de proyección y vista
            # projection = pyrr.matrix44.create_perspective_projection_matrix(
            #     45.0, self.width / self.height, 0.1, 100.0
            # )

            # view = pyrr.matrix44.create_look_at(
            #     np.array([0.0, 3.0, 15.0]),
            #     np.array([0.0, 0.0, 0.0]),
            #     np.array([0.0, 1.0, 0.0]),
            # )

            self.particle_system.render(self.projection, self.view)

            # Swap buffers y poll eventos
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        # Limpieza
        self.particle_system.cleanup()
        glfw.terminate()

    def update_mouse_velocity(self, current_pos):
        current_time = glfw.get_time()
        dt = current_time - self.last_time
        self.last_mouse_time = current_time

        if self.last_mouse_pos is not None and dt > 0:
            dx = current_pos[0] - self.last_mouse_pos[0]
            dy = current_pos[1] - self.last_mouse_pos[1]
            vx = dx / dt
            vy = dy / dt

            k = 0.8

            self.mouse_velocity[0] = (
                self.mouse_alpha * vx + (1 - self.mouse_alpha) * self.mouse_velocity[0]
            ) * k
            self.mouse_velocity[1] = (
                self.mouse_alpha * vy + (1 - self.mouse_alpha) * self.mouse_velocity[1]
            ) * k

        self.last_mouse_pos = current_pos

    def clamp_mouse_velocity(self, vmin=-1, vmax=1):
        MAX_VEL = 300
        MIN_VEL = -300

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

            self.update_mouse_velocity([x_pixel, y_pixel])

            self.mosue_particle_emitted = False
            if self.mouse_particle == None:
                self.mouse_particle = Particle()
            else:
                self.mouse_particle.position[0] = x_pixel
                self.mouse_particle.position[1] = y_pixel

        if glfw.get_mouse_button(self.window, glfw.MOUSE_BUTTON_LEFT) == glfw.RELEASE:
            if self.mouse_particle != None:
                self.clamp_mouse_velocity()

                self.mouse_particle.velocity[0] = self.mouse_velocity[0]
                self.mouse_particle.velocity[1] = self.mouse_velocity[1]

                self.particle_system.emit(
                    Particle(
                        color=self.mouse_particle.color,
                        velocity=self.mouse_particle.velocity,
                        position=self.mouse_particle.position,
                    ),
                )

                self.mouse_particle = None
                self.mosue_particle_emitted = True

        if glfw.get_key(self.window, glfw.KEY_R) == glfw.PRESS:
            self.particle_system.reset()


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
