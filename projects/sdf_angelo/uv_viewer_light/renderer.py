import ctypes

import numpy as np


class GLMeshRenderer:

    def __init__(self, mesh, texture_size, background=(0.08, 0.08, 0.10, 1.0)):
        gl_vertex_pointer_raw = None
        gl_texcoord_pointer_raw = None
        opengl_error_cls = RuntimeError
        try:
            from OpenGL.GL import (  # pylint: disable=import-error
                GL_ARRAY_BUFFER,
                GL_COLOR_BUFFER_BIT,
                GL_CULL_FACE,
                GL_DEPTH_BUFFER_BIT,
                GL_DEPTH_TEST,
                GL_ELEMENT_ARRAY_BUFFER,
                GL_FLOAT,
                GL_LINEAR,
                GL_MODELVIEW,
                GL_PROJECTION,
                GL_RGBA,
                GL_STATIC_DRAW,
                GL_TEXTURE_2D,
                GL_TEXTURE_MAG_FILTER,
                GL_TEXTURE_MIN_FILTER,
                GL_TRIANGLES,
                GL_TRUE,
                GL_UNPACK_ALIGNMENT,
                GL_UNSIGNED_BYTE,
                GL_UNSIGNED_INT,
                glBindBuffer,
                glBindTexture,
                glBufferData,
                glClear,
                glClearColor,
                glDeleteBuffers,
                glDeleteTextures,
                glDisable,
                glDrawElements,
                glEnable,
                glEnableClientState,
                glGenBuffers,
                glGenTextures,
                glLoadIdentity,
                glMatrixMode,
                glPixelStorei,
                glPolygonMode,
                glTexCoordPointer,
                glTexImage2D,
                glTexParameteri,
                glTexSubImage2D,
                glVertexPointer,
                glViewport,
                GL_FRONT_AND_BACK,
                GL_FILL,
                GL_LINE,
                GL_TEXTURE_COORD_ARRAY,
                GL_VERTEX_ARRAY,
            )
            from OpenGL.GLU import gluLookAt, gluPerspective  # pylint: disable=import-error
            from OpenGL import error as opengl_error  # pylint: disable=import-error
            opengl_error_cls = getattr(opengl_error, "Error", RuntimeError)
            try:
                from OpenGL.raw.GL.VERSION.GL_1_1 import (  # pylint: disable=import-error
                    glTexCoordPointer as glTexCoordPointerRaw,
                    glVertexPointer as glVertexPointerRaw,
                )
                gl_vertex_pointer_raw = glVertexPointerRaw
                gl_texcoord_pointer_raw = glTexCoordPointerRaw
            except Exception:
                gl_vertex_pointer_raw = None
                gl_texcoord_pointer_raw = None
        except Exception as exc:
            raise RuntimeError(
                "PyOpenGL is required for uv_viewer_light. Install PyOpenGL and PyOpenGL_accelerate."
            ) from exc

        self._gl = {
            name: value
            for name, value in locals().items()
            if name.startswith("GL_") or name.startswith("gl") or name.startswith("glu")
        }
        self.mesh = mesh
        self.texture_size = int(texture_size)
        self._wireframe = False
        self._opengl_error_cls = opengl_error_cls
        self._gl_vertex_pointer_raw = gl_vertex_pointer_raw
        self._gl_texcoord_pointer_raw = gl_texcoord_pointer_raw
        self._use_raw_pointer_calls = False

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.uint32).reshape(-1)
        uvs = np.asarray(mesh.visual.uv, dtype=np.float32).copy()
        if uvs.shape[0] != vertices.shape[0]:
            raise ValueError("UV mesh must have per-vertex UVs matching mesh vertices.")
        uvs[:, 1] = 1.0 - uvs[:, 1]
        interleaved = np.concatenate([vertices, uvs], axis=1).astype(np.float32, copy=False)

        self._index_count = int(faces.size)
        self._vbo = glGenBuffers(1)
        self._ebo = glGenBuffers(1)
        self._texture_id = glGenTextures(1)
        self._last_uploaded_update = -1

        glClearColor(*background)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_TEXTURE_2D)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, interleaved.nbytes, interleaved, GL_STATIC_DRAW)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces.nbytes, faces, GL_STATIC_DRAW)

        glBindTexture(GL_TEXTURE_2D, self._texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        empty = np.full((self.texture_size, self.texture_size, 4), 255, dtype=np.uint8)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            self.texture_size,
            self.texture_size,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            empty,
        )

        self._stride = 5 * 4
        self._vertex_offset = ctypes.c_void_p(0)
        self._uv_offset = ctypes.c_void_p(3 * 4)

    def shutdown(self):
        glDeleteBuffers = self._gl["glDeleteBuffers"]
        glDeleteTextures = self._gl["glDeleteTextures"]
        glDeleteBuffers(1, [self._vbo])
        glDeleteBuffers(1, [self._ebo])
        glDeleteTextures([self._texture_id])

    def resize(self, width, height):
        self._gl["glViewport"](0, 0, int(width), int(height))

    def toggle_wireframe(self):
        self._wireframe = not self._wireframe
        mode = self._gl["GL_LINE"] if self._wireframe else self._gl["GL_FILL"]
        self._gl["glPolygonMode"](self._gl["GL_FRONT_AND_BACK"], mode)

    def upload_texture(self, rgba_bytes):
        if rgba_bytes is None:
            return
        self._gl["glBindTexture"](self._gl["GL_TEXTURE_2D"], self._texture_id)
        self._gl["glTexSubImage2D"](
            self._gl["GL_TEXTURE_2D"],
            0,
            0,
            0,
            self.texture_size,
            self.texture_size,
            self._gl["GL_RGBA"],
            self._gl["GL_UNSIGNED_BYTE"],
            rgba_bytes,
        )

    def draw(self, camera, viewport_size):
        width = max(int(viewport_size[0]), 1)
        height = max(int(viewport_size[1]), 1)
        aspect = float(width) / float(height)
        _, up, _ = camera.get_basis()
        eye = camera.get_position()
        center = camera.target

        glClear = self._gl["glClear"]
        glMatrixMode = self._gl["glMatrixMode"]
        glLoadIdentity = self._gl["glLoadIdentity"]
        gluPerspective = self._gl["gluPerspective"]
        gluLookAt = self._gl["gluLookAt"]
        glBindBuffer = self._gl["glBindBuffer"]
        glBindTexture = self._gl["glBindTexture"]
        glEnableClientState = self._gl["glEnableClientState"]
        glVertexPointer = self._gl["glVertexPointer"]
        glTexCoordPointer = self._gl["glTexCoordPointer"]
        glDrawElements = self._gl["glDrawElements"]

        glClear(self._gl["GL_COLOR_BUFFER_BIT"] | self._gl["GL_DEPTH_BUFFER_BIT"])
        glMatrixMode(self._gl["GL_PROJECTION"])
        glLoadIdentity()
        gluPerspective(camera.fov_y_deg, aspect, camera.near, camera.far)
        glMatrixMode(self._gl["GL_MODELVIEW"])
        glLoadIdentity()
        gluLookAt(
            float(eye[0]), float(eye[1]), float(eye[2]),
            float(center[0]), float(center[1]), float(center[2]),
            float(up[0]), float(up[1]), float(up[2]),
        )

        glBindTexture(self._gl["GL_TEXTURE_2D"], self._texture_id)
        glBindBuffer(self._gl["GL_ARRAY_BUFFER"], self._vbo)
        glBindBuffer(self._gl["GL_ELEMENT_ARRAY_BUFFER"], self._ebo)
        glEnableClientState(self._gl["GL_VERTEX_ARRAY"])
        glEnableClientState(self._gl["GL_TEXTURE_COORD_ARRAY"])
        if self._use_raw_pointer_calls:
            self._gl_vertex_pointer_raw(3, self._gl["GL_FLOAT"], self._stride, self._vertex_offset)
            self._gl_texcoord_pointer_raw(2, self._gl["GL_FLOAT"], self._stride, self._uv_offset)
        else:
            try:
                glVertexPointer(3, self._gl["GL_FLOAT"], self._stride, self._vertex_offset)
                glTexCoordPointer(2, self._gl["GL_FLOAT"], self._stride, self._uv_offset)
            except self._opengl_error_cls as exc:
                can_fallback = (
                    self._gl_vertex_pointer_raw is not None
                    and self._gl_texcoord_pointer_raw is not None
                    and "no valid context" in str(exc).lower()
                )
                if not can_fallback:
                    raise
                self._use_raw_pointer_calls = True
                print(
                    "[uv_viewer_light] PyOpenGL context tracking failed in gl*Pointer; "
                    "falling back to raw GL calls."
                )
                self._gl_vertex_pointer_raw(3, self._gl["GL_FLOAT"], self._stride, self._vertex_offset)
                self._gl_texcoord_pointer_raw(2, self._gl["GL_FLOAT"], self._stride, self._uv_offset)
        glDrawElements(self._gl["GL_TRIANGLES"], self._index_count, self._gl["GL_UNSIGNED_INT"], None)
