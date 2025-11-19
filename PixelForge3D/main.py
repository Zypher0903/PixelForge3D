"""
Enhanced Professional 3D Modeling Engine v2.0
Improved GUI with modern styling, better layout, and bug fixes

Requirements:
pip install pygame PyOpenGL PyOpenGL_accelerate numpy pillow
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import json
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import os
from PIL import Image
import copy
import tkinter as tk
from tkinter import ttk, messagebox, colorchooser, filedialog
import threading

# ==================== Vector and Math Classes ====================

class Vector3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
    
    def to_list(self):
        return [self.x, self.y, self.z]
    
    def to_array(self):
        return np.array([self.x, self.y, self.z])
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return Vector3(self.x/length, self.y/length, self.z/length)
        return Vector3()
    
    def distance_to(self, other):
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __truediv__(self, scalar):
        if scalar != 0:
            return Vector3(self.x / scalar, self.y / scalar, self.z / scalar)
        return Vector3()
    
    def copy(self):
        return Vector3(self.x, self.y, self.z)
    
    def __repr__(self):
        return f"Vector3({self.x:.2f}, {self.y:.2f}, {self.z:.2f})"

# ==================== Material System ====================

class Material:
    def __init__(self, name="Material"):
        self.name = name
        self.color = [0.8, 0.8, 0.8]
        self.ambient = [0.2, 0.2, 0.2]
        self.diffuse = [0.8, 0.8, 0.8]
        self.specular = [1.0, 1.0, 1.0]
        self.shininess = 32.0
        self.metallic = 0.0
        self.roughness = 0.5
        self.emission = [0.0, 0.0, 0.0]
        self.wireframe = False
        self.shading_type = "smooth"
        self.texture_id = None
        self.use_texture = False
        self.texture_path = None
    
    def apply(self):
        glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, self.ambient + [1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, self.color + [1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, self.specular + [1.0])
        glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, self.emission + [1.0])
        glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, self.shininess)
        
        if self.use_texture and self.texture_id:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
        else:
            glDisable(GL_TEXTURE_2D)
    
    def create_procedural_texture(self, pattern="checker"):
        size = 256
        
        if pattern == "checker":
            data = []
            for i in range(size):
                for j in range(size):
                    c = 255 if ((i // 32) + (j // 32)) % 2 == 0 else 100
                    data.extend([c, c, c, 255])
        elif pattern == "grid":
            data = []
            for i in range(size):
                for j in range(size):
                    c = 50 if (i % 32 < 2 or j % 32 < 2) else 200
                    data.extend([c, c, c, 255])
        elif pattern == "gradient":
            data = []
            for i in range(size):
                for j in range(size):
                    c = int((i / size) * 255)
                    data.extend([c, c, c, 255])
        else:
            data = [255, 255, 255, 255] * (size * size)
        
        img_data = bytes(data)
        
        if self.texture_id:
            glDeleteTextures([self.texture_id])
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size, size,
                    0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
        
        self.use_texture = True
        self.texture_path = f"procedural_{pattern}"
    
    def copy(self):
        mat = Material(self.name)
        mat.color = self.color.copy()
        mat.ambient = self.ambient.copy()
        mat.diffuse = self.diffuse.copy()
        mat.specular = self.specular.copy()
        mat.shininess = self.shininess
        mat.metallic = self.metallic
        mat.roughness = self.roughness
        mat.emission = self.emission.copy()
        mat.wireframe = self.wireframe
        mat.shading_type = self.shading_type
        mat.texture_id = self.texture_id
        mat.use_texture = self.use_texture
        mat.texture_path = self.texture_path
        return mat

# ==================== Vertex Class ====================

class Vertex:
    def __init__(self, position, normal=None, uv=None):
        self.position = position if isinstance(position, Vector3) else Vector3(*position)
        self.normal = normal if normal else Vector3(0, 1, 0)
        self.uv = uv if uv else [0, 0]
        self.selected = False
        self.original_position = self.position.copy()
    
    def copy(self):
        v = Vertex(self.position.copy(), self.normal.copy(), self.uv.copy())
        v.selected = self.selected
        v.original_position = self.original_position.copy()
        return v

# ==================== Edge Class ====================

class Edge:
    def __init__(self, v1_idx, v2_idx):
        self.v1 = v1_idx
        self.v2 = v2_idx
        self.selected = False

# ==================== Face Class ====================

class Face:
    def __init__(self, vertex_indices):
        self.vertices = vertex_indices
        self.normal = Vector3(0, 1, 0)
        self.selected = False
    
    def calculate_normal(self, vertices):
        if len(self.vertices) >= 3:
            v1 = vertices[self.vertices[1]].position - vertices[self.vertices[0]].position
            v2 = vertices[self.vertices[2]].position - vertices[self.vertices[0]].position
            
            normal = v1.cross(v2)
            self.normal = normal.normalize()
        return self.normal
    
    def get_center(self, vertices):
        center = Vector3(0, 0, 0)
        for v_idx in self.vertices:
            center = center + vertices[v_idx].position
        return center / len(self.vertices)

# ==================== Mesh Class ====================

class Mesh:
    def __init__(self):
        self.vertices: List[Vertex] = []
        self.edges: List[Edge] = []
        self.faces: List[Face] = []
    
    def add_vertex(self, position, normal=None, uv=None):
        v = Vertex(position, normal, uv)
        self.vertices.append(v)
        return len(self.vertices) - 1
    
    def add_face(self, vertex_indices):
        face = Face(vertex_indices)
        self.faces.append(face)
        
        for i in range(len(vertex_indices)):
            v1 = vertex_indices[i]
            v2 = vertex_indices[(i + 1) % len(vertex_indices)]
            if not self.has_edge(v1, v2):
                self.edges.append(Edge(v1, v2))
        
        return len(self.faces) - 1
    
    def has_edge(self, v1, v2):
        for edge in self.edges:
            if (edge.v1 == v1 and edge.v2 == v2) or (edge.v1 == v2 and edge.v2 == v1):
                return True
        return False
    
    def calculate_normals(self):
        for vertex in self.vertices:
            vertex.normal = Vector3(0, 0, 0)
        
        for face in self.faces:
            face.calculate_normal(self.vertices)
            for v_idx in face.vertices:
                if v_idx < len(self.vertices):
                    self.vertices[v_idx].normal = self.vertices[v_idx].normal + face.normal
        
        for vertex in self.vertices:
            vertex.normal = vertex.normal.normalize()
    
    def subdivide(self):
        new_vertices = []
        new_faces = []
        
        face_points = []
        for face in self.faces:
            center = Vector3(0, 0, 0)
            for v_idx in face.vertices:
                center = center + self.vertices[v_idx].position
            center = center / len(face.vertices)
            face_points.append(center)
        
        vertex_offset = len(self.vertices)
        
        for i, face in enumerate(self.faces):
            face_center_idx = vertex_offset + i
            new_vertices.append(Vertex(face_points[i]))
            
            num_verts = len(face.vertices)
            for j in range(num_verts):
                v1 = face.vertices[j]
                v2 = face.vertices[(j + 1) % num_verts]
                
                edge_mid = (self.vertices[v1].position + self.vertices[v2].position) / 2
                edge_idx = len(self.vertices) + len(new_vertices)
                new_vertices.append(Vertex(edge_mid))
                
                prev_edge_idx = len(self.vertices) + len(face_points) + ((j - 1) % num_verts) * 2 + 1
                if j == 0:
                    prev_edge_idx = len(self.vertices) + len(face_points) + (num_verts - 1) * 2 + 1
                
                new_faces.append([v1, edge_idx, face_center_idx, prev_edge_idx])
        
        self.vertices.extend(new_vertices)
        self.faces = [Face(f) for f in new_faces if len(f) >= 3]
        self.rebuild_edges()
        self.calculate_normals()
    
    def rebuild_edges(self):
        self.edges.clear()
        for face in self.faces:
            for i in range(len(face.vertices)):
                v1 = face.vertices[i]
                v2 = face.vertices[(i + 1) % len(face.vertices)]
                if not self.has_edge(v1, v2):
                    self.edges.append(Edge(v1, v2))
    
    def extrude_selected_faces(self, distance=0.5):
        extruded_vertices = {}
        new_faces = []
        
        for face in self.faces:
            if face.selected:
                direction = face.normal * distance
                
                new_v_indices = []
                for v_idx in face.vertices:
                    if v_idx not in extruded_vertices:
                        old_pos = self.vertices[v_idx].position
                        new_pos = old_pos + direction
                        new_idx = self.add_vertex(new_pos)
                        extruded_vertices[v_idx] = new_idx
                    new_v_indices.append(extruded_vertices[v_idx])
                
                for i in range(len(face.vertices)):
                    v1 = face.vertices[i]
                    v2 = face.vertices[(i + 1) % len(face.vertices)]
                    nv1 = new_v_indices[i]
                    nv2 = new_v_indices[(i + 1) % len(new_v_indices)]
                    
                    new_faces.append(Face([v1, v2, nv2, nv1]))
                
                face.vertices = new_v_indices
        
        self.faces.extend(new_faces)
        self.rebuild_edges()
        self.calculate_normals()
    
    def bevel_selected_edges(self, amount=0.1):
        beveled_verts = {}
        new_faces = []
        
        for edge in self.edges:
            if edge.selected:
                v1_pos = self.vertices[edge.v1].position
                v2_pos = self.vertices[edge.v2].position
                
                new_v1_pos = v1_pos + (v2_pos - v1_pos) * amount
                new_v2_pos = v2_pos + (v1_pos - v2_pos) * amount
                
                nv1_idx = self.add_vertex(new_v1_pos)
                nv2_idx = self.add_vertex(new_v2_pos)
                
                beveled_verts[edge.v1] = nv1_idx
                beveled_verts[edge.v2] = nv2_idx
        
        self.rebuild_edges()
        self.calculate_normals()
    
    def inset_selected_faces(self, amount=0.2):
        for face in self.faces:
            if face.selected:
                center = face.get_center(self.vertices)
                
                new_v_indices = []
                for v_idx in face.vertices:
                    old_pos = self.vertices[v_idx].position
                    direction = center - old_pos
                    new_pos = old_pos + direction * amount
                    new_idx = self.add_vertex(new_pos)
                    new_v_indices.append(new_idx)
                
                for i in range(len(face.vertices)):
                    v1 = face.vertices[i]
                    v2 = face.vertices[(i + 1) % len(face.vertices)]
                    nv1 = new_v_indices[i]
                    nv2 = new_v_indices[(i + 1) % len(new_v_indices)]
                    
                    self.faces.append(Face([v1, v2, nv2, nv1]))
                
                face.vertices = new_v_indices
        
        self.rebuild_edges()
        self.calculate_normals()
    
    def copy(self):
        mesh = Mesh()
        mesh.vertices = [v.copy() for v in self.vertices]
        mesh.edges = [Edge(e.v1, e.v2) for e in self.edges]
        mesh.faces = [Face(f.vertices.copy()) for f in self.faces]
        mesh.calculate_normals()
        return mesh

# ==================== Object3D Class ====================

class Object3D:
    def __init__(self, name: str, object_type: str):
        self.name = name
        self.object_type = object_type
        self.position = Vector3(0, 0, 0)
        self.rotation = Vector3(0, 0, 0)
        self.scale = Vector3(1, 1, 1)
        self.material = Material()
        self.mesh = Mesh()
        self.selected = False
        self.visible = True
        self.display_list = None
        self.needs_update = True
        self.modifiers = []
        self.parent = None
        self.children = []
    
    def generate_geometry(self):
        if self.object_type == "cube":
            self._generate_cube()
        elif self.object_type == "sphere":
            self._generate_sphere(32, 32)
        elif self.object_type == "cylinder":
            self._generate_cylinder(32)
        elif self.object_type == "cone":
            self._generate_cone(32)
        elif self.object_type == "torus":
            self._generate_torus(32, 16)
        elif self.object_type == "plane":
            self._generate_plane(10, 10)
        elif self.object_type == "monkey":
            self._generate_monkey()
        
        self.mesh.calculate_normals()
        self.needs_update = True
    
    def _generate_cube(self):
        s = 0.5
        vertices = [
            Vector3(-s, -s, -s), Vector3(s, -s, -s), Vector3(s, s, -s), Vector3(-s, s, -s),
            Vector3(-s, -s, s), Vector3(s, -s, s), Vector3(s, s, s), Vector3(-s, s, s)
        ]
        
        for v in vertices:
            self.mesh.add_vertex(v)
        
        faces = [
            [0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1],
            [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]
        ]
        
        for face in faces:
            self.mesh.add_face(face)
    
    def _generate_sphere(self, rings, sectors):
        for i in range(rings + 1):
            lat = math.pi * (-0.5 + float(i) / rings)
            y = math.sin(lat) * 0.5
            r = math.cos(lat) * 0.5
            
            for j in range(sectors + 1):
                lng = 2 * math.pi * float(j) / sectors
                x = r * math.cos(lng)
                z = r * math.sin(lng)
                
                u = float(j) / sectors
                v = float(i) / rings
                
                self.mesh.add_vertex(Vector3(x, y, z), uv=[u, v])
        
        for i in range(rings):
            for j in range(sectors):
                first = i * (sectors + 1) + j
                second = first + sectors + 1
                
                self.mesh.add_face([first, second, first + 1])
                self.mesh.add_face([second, second + 1, first + 1])
    
    def _generate_cylinder(self, sectors):
        self.mesh.add_vertex(Vector3(0, 0.5, 0))
        self.mesh.add_vertex(Vector3(0, -0.5, 0))
        
        for i in range(sectors + 1):
            angle = 2 * math.pi * i / sectors
            x = math.cos(angle) * 0.5
            z = math.sin(angle) * 0.5
            
            u = float(i) / sectors
            
            self.mesh.add_vertex(Vector3(x, 0.5, z), uv=[u, 0])
            self.mesh.add_vertex(Vector3(x, -0.5, z), uv=[u, 1])
        
        for i in range(sectors):
            self.mesh.add_face([0, 2 + i * 2, 2 + ((i + 1) % sectors) * 2])
        
        for i in range(sectors):
            self.mesh.add_face([1, 3 + ((i + 1) % sectors) * 2, 3 + i * 2])
        
        for i in range(sectors):
            i1 = 2 + i * 2
            i2 = 2 + ((i + 1) % sectors) * 2
            self.mesh.add_face([i1, i1 + 1, i2 + 1, i2])
    
    def _generate_cone(self, sectors):
        self.mesh.add_vertex(Vector3(0, 0.5, 0))
        self.mesh.add_vertex(Vector3(0, -0.5, 0))
        
        for i in range(sectors + 1):
            angle = 2 * math.pi * i / sectors
            x = math.cos(angle) * 0.5
            z = math.sin(angle) * 0.5
            self.mesh.add_vertex(Vector3(x, -0.5, z))
        
        for i in range(sectors):
            self.mesh.add_face([0, 2 + i, 2 + ((i + 1) % sectors)])
        
        for i in range(sectors):
            self.mesh.add_face([1, 2 + ((i + 1) % sectors), 2 + i])
    
    def _generate_torus(self, major_segments, minor_segments):
        major_radius = 0.4
        minor_radius = 0.15
        
        for i in range(major_segments):
            theta = 2 * math.pi * i / major_segments
            
            for j in range(minor_segments):
                phi = 2 * math.pi * j / minor_segments
                
                x = (major_radius + minor_radius * math.cos(phi)) * math.cos(theta)
                y = minor_radius * math.sin(phi)
                z = (major_radius + minor_radius * math.cos(phi)) * math.sin(theta)
                
                u = float(i) / major_segments
                v = float(j) / minor_segments
                
                self.mesh.add_vertex(Vector3(x, y, z), uv=[u, v])
        
        for i in range(major_segments):
            for j in range(minor_segments):
                first = i * minor_segments + j
                second = ((i + 1) % major_segments) * minor_segments + j
                third = i * minor_segments + ((j + 1) % minor_segments)
                fourth = ((i + 1) % major_segments) * minor_segments + ((j + 1) % minor_segments)
                
                self.mesh.add_face([first, second, fourth, third])
    
    def _generate_plane(self, subdivisions_x=10, subdivisions_y=10):
        size = 2.0
        
        for i in range(subdivisions_y + 1):
            for j in range(subdivisions_x + 1):
                x = (j / subdivisions_x - 0.5) * size
                z = (i / subdivisions_y - 0.5) * size
                y = 0
                
                u = j / subdivisions_x
                v = i / subdivisions_y
                
                self.mesh.add_vertex(Vector3(x, y, z), uv=[u, v])
        
        for i in range(subdivisions_y):
            for j in range(subdivisions_x):
                v1 = i * (subdivisions_x + 1) + j
                v2 = v1 + 1
                v3 = v1 + (subdivisions_x + 1) + 1
                v4 = v1 + (subdivisions_x + 1)
                
                self.mesh.add_face([v1, v2, v3, v4])
    
    def _generate_monkey(self):
        self._generate_sphere(16, 16)
        
        for vertex in self.mesh.vertices:
            if vertex.position.z < 0:
                vertex.position.z *= 0.7
            vertex.position.y *= 1.2
    
    def build_display_list(self):
        if self.display_list:
            glDeleteLists(self.display_list, 1)
        
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)
        self._draw_mesh()
        glEndList()
        self.needs_update = False
    
    def _draw_mesh(self):
        if self.material.shading_type == "wireframe" or self.material.wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glDisable(GL_LIGHTING)
            glColor3f(*self.material.color)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
            self.material.apply()
        
        for face in self.mesh.faces:
            if len(face.vertices) == 3:
                glBegin(GL_TRIANGLES)
            elif len(face.vertices) == 4:
                glBegin(GL_QUADS)
            else:
                glBegin(GL_POLYGON)
            
            for v_idx in face.vertices:
                if v_idx < len(self.mesh.vertices):
                    vertex = self.mesh.vertices[v_idx]
                    
                    if self.material.shading_type == "smooth":
                        glNormal3f(*vertex.normal.to_list())
                    else:
                        glNormal3f(*face.normal.to_list())
                    
                    if self.material.use_texture:
                        glTexCoord2f(*vertex.uv)
                    
                    glVertex3f(*vertex.position.to_list())
            
            glEnd()
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    
    def draw(self, mode="object"):
        if not self.visible:
            return
        
        glPushMatrix()
        
        glTranslatef(self.position.x, self.position.y, self.position.z)
        glRotatef(self.rotation.x, 1, 0, 0)
        glRotatef(self.rotation.y, 0, 1, 0)
        glRotatef(self.rotation.z, 0, 0, 1)
        glScalef(self.scale.x, self.scale.y, self.scale.z)
        
        if mode == "object":
            if self.needs_update or not self.display_list:
                self.build_display_list()
            
            if self.display_list:
                glCallList(self.display_list)
        
        elif mode == "edit":
            self._draw_mesh()
            self._draw_edit_mode()
        
        if self.selected and mode == "object":
            glDisable(GL_LIGHTING)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(2.0)
            glColor3f(1.0, 0.6, 0.0)
            
            for face in self.mesh.faces:
                if len(face.vertices) == 3:
                    glBegin(GL_TRIANGLES)
                elif len(face.vertices) == 4:
                    glBegin(GL_QUADS)
                else:
                    glBegin(GL_POLYGON)
                
                for v_idx in face.vertices:
                    if v_idx < len(self.mesh.vertices):
                        glVertex3f(*self.mesh.vertices[v_idx].position.to_list())
                
                glEnd()
            
            glLineWidth(1.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
        
        glPopMatrix()
    
    def _draw_edit_mode(self):
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        
        glPointSize(8.0)
        glBegin(GL_POINTS)
        for vertex in self.mesh.vertices:
            if vertex.selected:
                glColor3f(1.0, 0.5, 0.0)
            else:
                glColor3f(0.0, 0.0, 0.0)
            glVertex3f(*vertex.position.to_list())
        glEnd()
        
        glLineWidth(2.0)
        glBegin(GL_LINES)
        for edge in self.mesh.edges:
            if edge.selected:
                glColor3f(1.0, 0.5, 0.0)
            else:
                glColor3f(0.0, 0.0, 0.0)
            
            if edge.v1 < len(self.mesh.vertices) and edge.v2 < len(self.mesh.vertices):
                v1 = self.mesh.vertices[edge.v1]
                v2 = self.mesh.vertices[edge.v2]
                glVertex3f(*v1.position.to_list())
                glVertex3f(*v2.position.to_list())
        glEnd()
        glLineWidth(1.0)
        
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        for face in self.mesh.faces:
            if face.selected:
                glColor4f(1.0, 0.5, 0.0, 0.3)
                
                if len(face.vertices) == 3:
                    glBegin(GL_TRIANGLES)
                elif len(face.vertices) == 4:
                    glBegin(GL_QUADS)
                else:
                    glBegin(GL_POLYGON)
                
                for v_idx in face.vertices:
                    if v_idx < len(self.mesh.vertices):
                        glVertex3f(*self.mesh.vertices[v_idx].position.to_list())
                
                glEnd()
        
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
    
    def copy(self):
        obj = Object3D(self.name + "_copy", self.object_type)
        obj.position = self.position.copy()
        obj.rotation = self.rotation.copy()
        obj.scale = self.scale.copy()
        obj.material = self.material.copy()
        obj.mesh = self.mesh.copy()
        obj.visible = self.visible
        return obj

# ==================== Camera Class ====================

class Camera:
    def __init__(self):
        self.position = Vector3(7, 7, 7)
        self.target = Vector3(0, 0, 0)
        self.up = Vector3(0, 1, 0)
        self.rotation_x = 45
        self.rotation_y = 45
        self.distance = 10.0
        self.fov = 60
        self.zoom_speed = 0.5
        self.rotation_speed = 0.3
        self.pan_speed = 0.01
        self.type = "perspective"
        self.ortho_scale = 5.0
        
        self.update()
    
    def update(self):
        rad_x = math.radians(self.rotation_x)
        rad_y = math.radians(self.rotation_y)
        
        self.position.x = self.target.x + self.distance * math.cos(rad_y) * math.cos(rad_x)
        self.position.y = self.target.y + self.distance * math.sin(rad_x)
        self.position.z = self.target.z + self.distance * math.sin(rad_y) * math.cos(rad_x)
    
    def apply(self, width, height):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        
        aspect = width / height if height > 0 else 1
        
        if self.type == "perspective":
            gluPerspective(self.fov, aspect, 0.1, 1000.0)
        else:
            scale = self.ortho_scale
            glOrtho(-scale * aspect, scale * aspect, -scale, scale, 0.1, 1000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            self.position.x, self.position.y, self.position.z,
            self.target.x, self.target.y, self.target.z,
            self.up.x, self.up.y, self.up.z
        )
    
    def rotate(self, dx, dy):
        self.rotation_y += dx * self.rotation_speed
        self.rotation_x += dy * self.rotation_speed
        self.rotation_x = max(-89, min(89, self.rotation_x))
        self.update()
    
    def zoom(self, delta):
        if self.type == "perspective":
            self.distance += delta * self.zoom_speed
            self.distance = max(1, min(100, self.distance))
        else:
            self.ortho_scale += delta * self.zoom_speed * 0.1
            self.ortho_scale = max(0.5, min(50, self.ortho_scale))
        self.update()
    
    def pan(self, dx, dy):
        forward = (self.target - self.position).normalize()
        right = forward.cross(self.up).normalize()
        
        move_right = right * dx * self.pan_speed * self.distance
        move_up = self.up * dy * self.pan_speed * self.distance
        
        self.target = self.target + move_right + move_up
        self.update()
    
    def reset(self):
        self.target = Vector3(0, 0, 0)
        self.rotation_x = 45
        self.rotation_y = 45
        self.distance = 10.0
        self.update()

# ==================== Light Class ====================

class Light:
    def __init__(self, light_id=GL_LIGHT0):
        self.light_id = light_id
        self.position = Vector3(10, 10, 5)
        self.ambient = [0.3, 0.3, 0.3, 1.0]
        self.diffuse = [0.8, 0.8, 0.8, 1.0]
        self.specular = [1.0, 1.0, 1.0, 1.0]
        self.enabled = True
    
    def apply(self):
        if self.enabled:
            glEnable(self.light_id)
            glLightfv(self.light_id, GL_POSITION, [self.position.x, self.position.y, self.position.z, 1.0])
            glLightfv(self.light_id, GL_AMBIENT, self.ambient)
            glLightfv(self.light_id, GL_DIFFUSE, self.diffuse)
            glLightfv(self.light_id, GL_SPECULAR, self.specular)
        else:
            glDisable(self.light_id)

# ==================== Scene Manager ====================

class SceneManager:
    def __init__(self):
        self.objects: List[Object3D] = []
        self.lights: List[Light] = []
        self.selected_objects: List[Object3D] = []
        self.object_counter = 0
        
        self.add_light()
    
    def add_object(self, object_type: str) -> Object3D:
        name = f"{object_type}_{self.object_counter}"
        obj = Object3D(name, object_type)
        obj.generate_geometry()
        self.objects.append(obj)
        self.object_counter += 1
        return obj
    
    def add_light(self) -> Light:
        light_id = GL_LIGHT0 + len(self.lights)
        if len(self.lights) < 8:
            light = Light(light_id)
            self.lights.append(light)
            return light
        return None
    
    def delete_object(self, obj: Object3D):
        if obj in self.objects:
            if obj.display_list:
                try:
                    glDeleteLists(obj.display_list, 1)
                except:
                    pass
            self.objects.remove(obj)
            if obj in self.selected_objects:
                self.selected_objects.remove(obj)
    
    def delete_selected(self):
        for obj in self.selected_objects[:]:
            self.delete_object(obj)
        self.selected_objects.clear()
    
    def select_object(self, obj: Object3D, multi=False):
        if not multi:
            for o in self.selected_objects:
                o.selected = False
            self.selected_objects.clear()
        
        if obj not in self.selected_objects:
            obj.selected = True
            self.selected_objects.append(obj)
        else:
            obj.selected = False
            self.selected_objects.remove(obj)
    
    def clear_selection(self):
        for obj in self.selected_objects:
            obj.selected = False
        self.selected_objects.clear()
    
    def duplicate_selected(self):
        new_objects = []
        for obj in self.selected_objects:
            new_obj = obj.copy()
            new_obj.name = f"{obj.name}_copy_{self.object_counter}"
            self.object_counter += 1
            new_obj.position = obj.position + Vector3(1, 0, 0)
            self.objects.append(new_obj)
            new_objects.append(new_obj)
        
        self.clear_selection()
        for obj in new_objects:
            self.select_object(obj, multi=True)
    
    def get_selected_object(self) -> Optional[Object3D]:
        return self.selected_objects[0] if self.selected_objects else None

# ==================== Enhanced Tkinter GUI ====================

class TkinterGUI:
    def __init__(self, engine):
        self.engine = engine
        self.root = tk.Tk()
        self.root.title("3D Modeling Engine v2.0 - Control Panel")
        self.root.geometry("450x850")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Configure style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        bg_color = '#2b2b2b'
        fg_color = '#e0e0e0'
        select_color = '#3d6fb5'
        
        style.configure('TNotebook', background=bg_color, borderwidth=0)
        style.configure('TNotebook.Tab', background='#3c3c3c', foreground=fg_color, 
                       padding=[10, 5], borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', select_color)])
        style.configure('TFrame', background=bg_color)
        style.configure('TLabel', background=bg_color, foreground=fg_color)
        style.configure('TButton', background='#3c3c3c', foreground=fg_color, borderwidth=1)
        style.map('TButton', background=[('active', select_color)])
        style.configure('TLabelframe', background=bg_color, foreground=fg_color)
        style.configure('TLabelframe.Label', background=bg_color, foreground=fg_color)
        
        self.root.configure(bg=bg_color)
        
        # Create menu bar
        self.create_menu()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Create tabs
        self.create_hierarchy_tab()
        self.create_properties_tab()
        self.create_tools_tab()
        self.create_material_tab()
        self.create_viewport_tab()
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready | Mode: Object")
        status_frame = tk.Frame(self.root, bg='#1e1e1e', relief=tk.SUNKEN, bd=1)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_bar = tk.Label(status_frame, textvariable=self.status_var, 
                                   bg='#1e1e1e', fg='#e0e0e0', anchor=tk.W, 
                                   font=('Consolas', 9))
        self.status_bar.pack(fill=tk.X, padx=5, pady=2)
        
        self.update_timer = None
        self.start_update_loop()
    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Scene", command=self.new_scene)
        file_menu.add_command(label="Save Scene (Ctrl+S)", command=self.engine.save_scene)
        file_menu.add_command(label="Load Scene (Ctrl+O)", command=self.engine.load_scene)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        
        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo (Ctrl+Z)", command=self.engine.undo)
        edit_menu.add_command(label="Redo (Shift+Ctrl+Z)", command=self.engine.redo)
        edit_menu.add_separator()
        edit_menu.add_command(label="Duplicate (Shift+D)", command=self.duplicate_object)
        edit_menu.add_command(label="Delete (X)", command=self.delete_object)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Reset Camera (Home)", command=self.engine.camera.reset)
        view_menu.add_checkbutton(label="Show Grid", variable=tk.BooleanVar(value=True),
                                  command=self.toggle_grid)
        view_menu.add_checkbutton(label="Show Axes", variable=tk.BooleanVar(value=True),
                                  command=self.toggle_axes)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Keyboard Shortcuts", command=self.show_shortcuts)
        help_menu.add_command(label="About", command=self.show_about)
    
    def create_hierarchy_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="⚙ Hierarchy")
        
        # Add object buttons with better layout
        add_frame = ttk.LabelFrame(frame, text="Add Primitive Object")
        add_frame.pack(fill='x', padx=5, pady=5)
        
        objects = [
            ("📦 Cube", "cube"),
            ("⚪ Sphere", "sphere"),
            ("🔵 Cylinder", "cylinder"),
            ("🔺 Cone", "cone"),
            ("⭕ Torus", "torus"),
            ("▭ Plane", "plane"),
            ("🐵 Monkey", "monkey")
        ]
        
        for i, (label, obj_type) in enumerate(objects):
            btn = ttk.Button(add_frame, text=label, width=20,
                           command=lambda t=obj_type: self.engine.add_object(t))
            btn.grid(row=i//2, column=i%2, padx=3, pady=3, sticky='ew')
        
        add_frame.columnconfigure(0, weight=1)
        add_frame.columnconfigure(1, weight=1)
        
        # Object list with search
        list_frame = ttk.LabelFrame(frame, text="Scene Objects")
        list_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Search box
        search_frame = ttk.Frame(list_frame)
        search_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(search_frame, text="🔍 Search:").pack(side=tk.LEFT, padx=2)
        self.search_var = tk.StringVar()
        self.search_var.trace('w', self.filter_objects)
        search_entry = ttk.Entry(search_frame, textvariable=self.search_var)
        search_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=2)
        
        # Scrollbar and listbox
        list_container = ttk.Frame(list_frame)
        list_container.pack(fill='both', expand=True, padx=5)
        
        scrollbar = ttk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.object_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set,
                                        bg='#3c3c3c', fg='#e0e0e0', 
                                        selectbackground='#3d6fb5',
                                        font=('Consolas', 10))
        self.object_listbox.pack(fill='both', expand=True)
        scrollbar.config(command=self.object_listbox.yview)
        
        self.object_listbox.bind('<<ListboxSelect>>', self.on_object_select)
        self.object_listbox.bind('<Double-Button-1>', self.focus_selected)
        
        # Object operations with icons
        ops_frame = ttk.Frame(frame)
        ops_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(ops_frame, text="📋 Duplicate", width=12,
                  command=self.duplicate_object).pack(side=tk.LEFT, padx=2)
        ttk.Button(ops_frame, text="🗑️ Delete", width=12,
                  command=self.delete_object).pack(side=tk.LEFT, padx=2)
        ttk.Button(ops_frame, text="👁️ Toggle", width=12,
                  command=self.toggle_visibility).pack(side=tk.LEFT, padx=2)
    
    def create_properties_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="📐 Properties")
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(frame, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Object name
        name_frame = ttk.LabelFrame(scrollable_frame, text="Object Name")
        name_frame.pack(fill='x', padx=5, pady=5)
        
        self.name_var = tk.StringVar(value="")
        name_entry = ttk.Entry(name_frame, textvariable=self.name_var)
        name_entry.pack(fill='x', padx=5, pady=5)
        ttk.Button(name_frame, text="Rename", command=self.rename_object).pack(pady=2)
        
        # Transform section
        transform_frame = ttk.LabelFrame(scrollable_frame, text="Transform")
        transform_frame.pack(fill='x', padx=5, pady=5)
        
        # Position
        pos_frame = ttk.Frame(transform_frame)
        pos_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(pos_frame, text="📍 Position:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, columnspan=6, sticky='w', pady=(0,3))
        
        self.pos_x_var = tk.DoubleVar(value=0)
        self.pos_y_var = tk.DoubleVar(value=0)
        self.pos_z_var = tk.DoubleVar(value=0)
        
        ttk.Label(pos_frame, text="X:").grid(row=1, column=0, padx=2)
        ttk.Entry(pos_frame, textvariable=self.pos_x_var, width=8).grid(row=1, column=1, padx=2)
        ttk.Label(pos_frame, text="Y:").grid(row=1, column=2, padx=2)
        ttk.Entry(pos_frame, textvariable=self.pos_y_var, width=8).grid(row=1, column=3, padx=2)
        ttk.Label(pos_frame, text="Z:").grid(row=1, column=4, padx=2)
        ttk.Entry(pos_frame, textvariable=self.pos_z_var, width=8).grid(row=1, column=5, padx=2)
        
        # Rotation
        rot_frame = ttk.Frame(transform_frame)
        rot_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(rot_frame, text="🔄 Rotation:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, columnspan=6, sticky='w', pady=(5,3))
        
        self.rot_x_var = tk.DoubleVar(value=0)
        self.rot_y_var = tk.DoubleVar(value=0)
        self.rot_z_var = tk.DoubleVar(value=0)
        
        ttk.Label(rot_frame, text="X:").grid(row=1, column=0, padx=2)
        ttk.Entry(rot_frame, textvariable=self.rot_x_var, width=8).grid(row=1, column=1, padx=2)
        ttk.Label(rot_frame, text="Y:").grid(row=1, column=2, padx=2)
        ttk.Entry(rot_frame, textvariable=self.rot_y_var, width=8).grid(row=1, column=3, padx=2)
        ttk.Label(rot_frame, text="Z:").grid(row=1, column=4, padx=2)
        ttk.Entry(rot_frame, textvariable=self.rot_z_var, width=8).grid(row=1, column=5, padx=2)
        
        # Scale
        scale_frame = ttk.Frame(transform_frame)
        scale_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(scale_frame, text="📏 Scale:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, columnspan=6, sticky='w', pady=(5,3))
        
        self.scale_x_var = tk.DoubleVar(value=1)
        self.scale_y_var = tk.DoubleVar(value=1)
        self.scale_z_var = tk.DoubleVar(value=1)
        
        ttk.Label(scale_frame, text="X:").grid(row=1, column=0, padx=2)
        ttk.Entry(scale_frame, textvariable=self.scale_x_var, width=8).grid(row=1, column=1, padx=2)
        ttk.Label(scale_frame, text="Y:").grid(row=1, column=2, padx=2)
        ttk.Entry(scale_frame, textvariable=self.scale_y_var, width=8).grid(row=1, column=3, padx=2)
        ttk.Label(scale_frame, text="Z:").grid(row=1, column=4, padx=2)
        ttk.Entry(scale_frame, textvariable=self.scale_z_var, width=8).grid(row=1, column=5, padx=2)
        
        ttk.Button(transform_frame, text="✓ Apply Transform", 
                  command=self.apply_transform).pack(pady=10, fill='x', padx=5)
        
        # Mesh Info
        info_frame = ttk.LabelFrame(scrollable_frame, text="Mesh Information")
        info_frame.pack(fill='x', padx=5, pady=5)
        
        self.mesh_info_var = tk.StringVar(value="No object selected")
        info_label = tk.Label(info_frame, textvariable=self.mesh_info_var, 
                            justify=tk.LEFT, bg='#2b2b2b', fg='#e0e0e0',
                            font=('Consolas', 9))
        info_label.pack(padx=10, pady=10, anchor='w')
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_tools_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🔧 Tools")
        
        # Mode selection
        mode_frame = ttk.LabelFrame(frame, text="Interaction Mode")
        mode_frame.pack(fill='x', padx=5, pady=5)
        
        self.mode_var = tk.StringVar(value="object")
        mode_btn_frame = ttk.Frame(mode_frame)
        mode_btn_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Radiobutton(mode_btn_frame, text="🖱️ Object Mode", variable=self.mode_var, 
                       value="object", command=self.change_mode).pack(anchor='w', padx=10, pady=2)
        ttk.Radiobutton(mode_btn_frame, text="✏️ Edit Mode (Tab)", variable=self.mode_var, 
                       value="edit", command=self.change_mode).pack(anchor='w', padx=10, pady=2)
        
        # Transform tools
        transform_frame = ttk.LabelFrame(frame, text="Transform Tools")
        transform_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(transform_frame, text="↔️ Grab (G)", 
                  command=lambda: self.engine.start_transform_tool("grab")).pack(
                      fill='x', padx=5, pady=2)
        ttk.Button(transform_frame, text="🔄 Rotate (R)", 
                  command=lambda: self.engine.start_transform_tool("rotate")).pack(
                      fill='x', padx=5, pady=2)
        ttk.Button(transform_frame, text="📏 Scale (S)", 
                  command=lambda: self.engine.start_transform_tool("scale")).pack(
                      fill='x', padx=5, pady=2)
        
        ttk.Label(transform_frame, text="💡 Tip: Press X, Y, or Z to constrain axis",
                 font=('Arial', 8, 'italic')).pack(padx=5, pady=5)
        
        # Edit mode tools
        self.edit_frame = ttk.LabelFrame(frame, text="Edit Mode Tools")
        self.edit_frame.pack(fill='x', padx=5, pady=5)
        
        self.edit_mode_type_var = tk.StringVar(value="vertex")
        
        selection_frame = ttk.Frame(self.edit_frame)
        selection_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Radiobutton(selection_frame, text="⚫ Vertex (1)", variable=self.edit_mode_type_var, 
                       value="vertex", command=self.change_edit_mode_type).pack(anchor='w', padx=10)
        ttk.Radiobutton(selection_frame, text="➖ Edge (2)", variable=self.edit_mode_type_var, 
                       value="edge", command=self.change_edit_mode_type).pack(anchor='w', padx=10)
        ttk.Radiobutton(selection_frame, text="⬜ Face (3)", variable=self.edit_mode_type_var, 
                       value="face", command=self.change_edit_mode_type).pack(anchor='w', padx=10)
        
        ttk.Separator(self.edit_frame, orient='horizontal').pack(fill='x', pady=5)
        
        # Modeling operations
        ops_label = ttk.Label(self.edit_frame, text="Modeling Operations:", 
                             font=('Arial', 9, 'bold'))
        ops_label.pack(anchor='w', padx=5, pady=(5,2))
        
        # Extrude
        extrude_frame = ttk.Frame(self.edit_frame)
        extrude_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(extrude_frame, text="Distance:").pack(side=tk.LEFT, padx=2)
        self.extrude_var = tk.DoubleVar(value=0.5)
        extrude_scale = ttk.Scale(extrude_frame, from_=0.1, to=2.0, 
                                 variable=self.extrude_var, orient=tk.HORIZONTAL)
        extrude_scale.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
        ttk.Label(extrude_frame, textvariable=self.extrude_var, width=5).pack(side=tk.LEFT)
        
        ttk.Button(self.edit_frame, text="↗️ Extrude (E)", 
                  command=self.extrude_faces).pack(fill='x', padx=5, pady=2)
        
        # Inset
        inset_frame = ttk.Frame(self.edit_frame)
        inset_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(inset_frame, text="Amount:").pack(side=tk.LEFT, padx=2)
        self.inset_var = tk.DoubleVar(value=0.2)
        inset_scale = ttk.Scale(inset_frame, from_=0.05, to=0.5, 
                               variable=self.inset_var, orient=tk.HORIZONTAL)
        inset_scale.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
        ttk.Label(inset_frame, textvariable=self.inset_var, width=5).pack(side=tk.LEFT)
        
        ttk.Button(self.edit_frame, text="⬇️ Inset (I)", 
                  command=self.inset_faces).pack(fill='x', padx=5, pady=2)
        
        # Bevel
        bevel_frame = ttk.Frame(self.edit_frame)
        bevel_frame.pack(fill='x', padx=5, pady=2)
        ttk.Label(bevel_frame, text="Amount:").pack(side=tk.LEFT, padx=2)
        self.bevel_var = tk.DoubleVar(value=0.1)
        bevel_scale = ttk.Scale(bevel_frame, from_=0.01, to=0.3, 
                               variable=self.bevel_var, orient=tk.HORIZONTAL)
        bevel_scale.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
        ttk.Label(bevel_frame, textvariable=self.bevel_var, width=5).pack(side=tk.LEFT)
        
        ttk.Button(self.edit_frame, text="🔷 Bevel (B)", 
                  command=self.bevel_edges).pack(fill='x', padx=5, pady=2)
        
        ttk.Separator(self.edit_frame, orient='horizontal').pack(fill='x', pady=5)
        
        ttk.Button(self.edit_frame, text="✂️ Subdivide (Ctrl+R)", 
                  command=self.subdivide_mesh).pack(fill='x', padx=5, pady=2)
        ttk.Button(self.edit_frame, text="☑️ Select All (A)", 
                  command=self.engine.toggle_select_all).pack(fill='x', padx=5, pady=2)
        
        # File operations
        file_frame = ttk.LabelFrame(frame, text="File Operations")
        file_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(file_frame, text="💾 Save Scene", 
                  command=self.engine.save_scene).pack(fill='x', padx=5, pady=2)
        ttk.Button(file_frame, text="📂 Load Scene", 
                  command=self.engine.load_scene).pack(fill='x', padx=5, pady=2)
        
        # Camera
        camera_frame = ttk.LabelFrame(frame, text="Camera Controls")
        camera_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(camera_frame, text="🏠 Reset Camera (Home)", 
                  command=self.engine.camera.reset).pack(fill='x', padx=5, pady=2)
        
        cam_type_frame = ttk.Frame(camera_frame)
        cam_type_frame.pack(fill='x', padx=5, pady=5)
        
        self.cam_type_var = tk.StringVar(value="perspective")
        ttk.Radiobutton(cam_type_frame, text="📷 Perspective", variable=self.cam_type_var,
                       value="perspective", command=self.change_camera_type).pack(anchor='w', padx=10)
        ttk.Radiobutton(cam_type_frame, text="📐 Orthographic", variable=self.cam_type_var,
                       value="orthographic", command=self.change_camera_type).pack(anchor='w', padx=10)
    
    def create_material_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="🎨 Material")
        
        # Create canvas with scrollbar
        canvas = tk.Canvas(frame, bg='#2b2b2b', highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Material properties
        mat_frame = ttk.LabelFrame(scrollable_frame, text="Material Properties")
        mat_frame.pack(fill='x', padx=5, pady=5)
        
        # Color
        color_frame = ttk.Frame(mat_frame)
        color_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(color_frame, text="🎨 Base Color:").pack(side=tk.LEFT, padx=2)
        self.color_canvas = tk.Canvas(color_frame, width=30, height=20, 
                                     bg='#cccccc', highlightthickness=1,
                                     highlightbackground='#666')
        self.color_canvas.pack(side=tk.LEFT, padx=5)
        self.color_button = ttk.Button(color_frame, text="Choose Color", 
                                       command=self.choose_color)
        self.color_button.pack(side=tk.LEFT, padx=5)
        self.current_color = [0.8, 0.8, 0.8]
        
        # Shininess
        shin_frame = ttk.Frame(mat_frame)
        shin_frame.pack(fill='x', padx=5, pady=5)
        ttk.Label(shin_frame, text="✨ Shininess:").pack(side=tk.LEFT, padx=2)
        self.shininess_var = tk.DoubleVar(value=32.0)
        shin_scale = ttk.Scale(shin_frame, from_=1.0, to=128.0, 
                              variable=self.shininess_var, orient=tk.HORIZONTAL)
        shin_scale.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
        ttk.Label(shin_frame, textvariable=self.shininess_var, width=6).pack(side=tk.LEFT)
        
        # Wireframe
        self.wireframe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mat_frame, text="📊 Wireframe Mode", 
                       variable=self.wireframe_var, command=self.toggle_wireframe).pack(
                           anchor='w', padx=5, pady=5)
        
        # Shading type
        shading_frame = ttk.LabelFrame(mat_frame, text="Shading Type")
        shading_frame.pack(fill='x', padx=5, pady=5)
        
        self.shading_var = tk.StringVar(value="smooth")
        ttk.Radiobutton(shading_frame, text="🌊 Smooth Shading", variable=self.shading_var, 
                       value="smooth", command=self.change_shading).pack(anchor='w', padx=10, pady=2)
        ttk.Radiobutton(shading_frame, text="📐 Flat Shading", variable=self.shading_var, 
                       value="flat", command=self.change_shading).pack(anchor='w', padx=10, pady=2)
        
        # Procedural textures
        texture_frame = ttk.LabelFrame(scrollable_frame, text="Procedural Textures")
        texture_frame.pack(fill='x', padx=5, pady=5)
        
        tex_btn_frame = ttk.Frame(texture_frame)
        tex_btn_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(tex_btn_frame, text="⬛⬜ Checker", 
                  command=lambda: self.apply_texture("checker")).pack(
                      fill='x', pady=2)
        ttk.Button(tex_btn_frame, text="# Grid", 
                  command=lambda: self.apply_texture("grid")).pack(
                      fill='x', pady=2)
        ttk.Button(tex_btn_frame, text="🌈 Gradient", 
                  command=lambda: self.apply_texture("gradient")).pack(
                      fill='x', pady=2)
        
        self.use_texture_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(texture_frame, text="✓ Use Texture", 
                       variable=self.use_texture_var, command=self.toggle_texture).pack(
                           anchor='w', padx=5, pady=5)
        
        ttk.Button(mat_frame, text="✓ Apply Material", 
                  command=self.apply_material).pack(fill='x', padx=5, pady=10)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_viewport_tab(self):
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="👁️ Viewport")
        
        # Display settings
        display_frame = ttk.LabelFrame(frame, text="Display Settings")
        display_frame.pack(fill='x', padx=5, pady=5)
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="# Show Grid", 
                       variable=self.show_grid_var, 
                       command=self.toggle_grid).pack(anchor='w', padx=10, pady=2)
        
        self.show_axes_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="📏 Show Axes", 
                       variable=self.show_axes_var, 
                       command=self.toggle_axes).pack(anchor='w', padx=10, pady=2)
        
        # Background color
        bg_frame = ttk.LabelFrame(frame, text="Background Color")
        bg_frame.pack(fill='x', padx=5, pady=5)
        
        bg_presets = [
            ("Dark Gray", (0.18, 0.18, 0.19)),
            ("Light Gray", (0.5, 0.5, 0.5)),
            ("Black", (0.0, 0.0, 0.0)),
            ("White", (1.0, 1.0, 1.0)),
            ("Sky Blue", (0.53, 0.81, 0.92))
        ]
        
        for name, color in bg_presets:
            ttk.Button(bg_frame, text=name, 
                      command=lambda c=color: self.set_background(c)).pack(
                          fill='x', padx=5, pady=2)
        
        # Statistics
        stats_frame = ttk.LabelFrame(frame, text="Performance Statistics")
        stats_frame.pack(fill='x', padx=5, pady=5)
        
        self.fps_var = tk.StringVar(value="FPS: 60")
        ttk.Label(stats_frame, textvariable=self.fps_var, 
                 font=('Consolas', 10)).pack(padx=10, pady=5, anchor='w')
        
        self.draw_calls_var = tk.StringVar(value="Draw Calls: 0")
        ttk.Label(stats_frame, textvariable=self.draw_calls_var,
                 font=('Consolas', 10)).pack(padx=10, pady=5, anchor='w')
        
        # Quick actions
        quick_frame = ttk.LabelFrame(frame, text="Quick Actions")
        quick_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Button(quick_frame, text="🖼️ Screenshot (F12)", 
                  command=self.take_screenshot).pack(fill='x', padx=5, pady=2)
        ttk.Button(quick_frame, text="🔄 Reload Scene", 
                  command=self.reload_scene).pack(fill='x', padx=5, pady=2)
        ttk.Button(quick_frame, text="🗑️ Clear All Objects", 
                  command=self.clear_scene).pack(fill='x', padx=5, pady=2)
    
    # Callback methods
    def new_scene(self):
        if messagebox.askyesno("New Scene", "Create new scene? Unsaved changes will be lost."):
            self.engine.scene.objects.clear()
            self.engine.scene.selected_objects.clear()
            self.engine.scene.object_counter = 0
            self.engine.history.clear()
            self.engine.history_index = -1
            self.update_hierarchy()
    
    def on_object_select(self, event):
        selection = self.object_listbox.curselection()
        if selection:
            idx = selection[0]
            # Find actual object from filtered list
            display_objects = self.get_filtered_objects()
            if idx < len(display_objects):
                obj = display_objects[idx]
                self.engine.scene.clear_selection()
                self.engine.scene.select_object(obj)
                self.update_properties()
    
    def get_filtered_objects(self):
        search_term = self.search_var.get().lower()
        if not search_term:
            return self.engine.scene.objects
        return [obj for obj in self.engine.scene.objects if search_term in obj.name.lower()]
    
    def filter_objects(self, *args):
        self.update_hierarchy()
    
    def focus_selected(self, event):
        obj = self.engine.scene.get_selected_object()
        if obj:
            self.engine.camera.target = obj.position.copy()
            self.engine.camera.update()
    
    def rename_object(self):
        obj = self.engine.scene.get_selected_object()
        if obj:
            new_name = self.name_var.get().strip()
            if new_name:
                obj.name = new_name
                self.update_hierarchy()
    
    def duplicate_object(self):
        self.engine.save_history_state()
        self.engine.scene.duplicate_selected()
        self.update_hierarchy()
    
    def delete_object(self):
        if not self.engine.scene.selected_objects:
            return
        
        count = len(self.engine.scene.selected_objects)
        if messagebox.askyesno("Delete", f"Delete {count} object(s)?"):
            self.engine.save_history_state()
            if self.engine.mode == "edit":
                self.engine.delete_selected_elements()
            else:
                self.engine.scene.delete_selected()
            self.update_hierarchy()
    
    def toggle_visibility(self):
        obj = self.engine.scene.get_selected_object()
        if obj:
            obj.visible = not obj.visible
            self.update_hierarchy()
    
    def apply_transform(self):
        obj = self.engine.scene.get_selected_object()
        if obj:
            try:
                obj.position.x = self.pos_x_var.get()
                obj.position.y = self.pos_y_var.get()
                obj.position.z = self.pos_z_var.get()
                obj.rotation.x = self.rot_x_var.get()
                obj.rotation.y = self.rot_y_var.get()
                obj.rotation.z = self.rot_z_var.get()
                obj.scale.x = max(0.01, self.scale_x_var.get())
                obj.scale.y = max(0.01, self.scale_y_var.get())
                obj.scale.z = max(0.01, self.scale_z_var.get())
                obj.needs_update = True
            except tk.TclError:
                messagebox.showerror("Error", "Invalid number format")
    
    def change_mode(self):
        self.engine.mode = self.mode_var.get()
        self.update_status()
    
    def change_edit_mode_type(self):
        self.engine.edit_mode_type = self.edit_mode_type_var.get()
        self.update_status()
    
    def change_camera_type(self):
        self.engine.camera.type = self.cam_type_var.get()
    
    def extrude_faces(self):
        self.engine.save_history_state()
        obj = self.engine.scene.get_selected_object()
        if obj and self.engine.edit_mode_type == "face":
            obj.mesh.extrude_selected_faces(self.extrude_var.get())
            obj.needs_update = True
    
    def inset_faces(self):
        self.engine.save_history_state()
        obj = self.engine.scene.get_selected_object()
        if obj and self.engine.edit_mode_type == "face":
            obj.mesh.inset_selected_faces(self.inset_var.get())
            obj.needs_update = True
    
    def bevel_edges(self):
        self.engine.save_history_state()
        obj = self.engine.scene.get_selected_object()
        if obj and self.engine.edit_mode_type == "edge":
            obj.mesh.bevel_selected_edges(self.bevel_var.get())
            obj.needs_update = True
    
    def subdivide_mesh(self):
        self.engine.save_history_state()
        obj = self.engine.scene.get_selected_object()
        if obj:
            obj.mesh.subdivide()
            obj.needs_update = True
            self.update_properties()
    
    def choose_color(self):
        obj = self.engine.scene.get_selected_object()
        if obj:
            current = tuple(int(c * 255) for c in obj.material.color)
            color = colorchooser.askcolor(color=current, title="Choose Color")
            if color[0]:
                self.current_color = [c / 255.0 for c in color[0]]
                obj.material.color = self.current_color
                obj.material.diffuse = self.current_color
                obj.needs_update = True
                self.update_color_preview()
    
    def update_color_preview(self):
        color_hex = '#%02x%02x%02x' % tuple(int(c * 255) for c in self.current_color)
        self.color_canvas.config(bg=color_hex)
    
    def toggle_wireframe(self):
        obj = self.engine.scene.get_selected_object()
        if obj:
            obj.material.wireframe = self.wireframe_var.get()
            obj.needs_update = True
    
    def change_shading(self):
        obj = self.engine.scene.get_selected_object()
        if obj:
            obj.material.shading_type = self.shading_var.get()
            obj.needs_update = True
    
    def apply_texture(self, pattern):
        obj = self.engine.scene.get_selected_object()
        if obj:
            obj.material.create_procedural_texture(pattern)
            obj.needs_update = True
            self.use_texture_var.set(True)
    
    def toggle_texture(self):
        obj = self.engine.scene.get_selected_object()
        if obj:
            obj.material.use_texture = self.use_texture_var.get()
            obj.needs_update = True
    
    def apply_material(self):
        obj = self.engine.scene.get_selected_object()
        if obj:
            obj.material.shininess = self.shininess_var.get()
            obj.material.color = self.current_color.copy()
            obj.material.diffuse = self.current_color.copy()
            obj.needs_update = True
    
    def toggle_grid(self):
        self.engine.show_grid = self.show_grid_var.get()
    
    def toggle_axes(self):
        self.engine.show_axes = self.show_axes_var.get()
    
    def set_background(self, color):
        self.engine.background_color = color
        glClearColor(*color, 1)
    
    def take_screenshot(self):
        self.engine.take_screenshot()
    
    def reload_scene(self):
        if messagebox.askyesno("Reload", "Reload scene from file?"):
            self.engine.load_scene()
            self.update_hierarchy()
    
    def clear_scene(self):
        if messagebox.askyesno("Clear Scene", "Delete all objects?"):
            self.engine.save_history_state()
            self.engine.scene.objects.clear()
            self.engine.scene.selected_objects.clear()
            self.update_hierarchy()
    
    def show_shortcuts(self):
        shortcuts = """
        KEYBOARD SHORTCUTS:
        
        Navigation:
        • Mouse Drag - Rotate view
        • Shift + Mouse - Pan view
        • Mouse Wheel - Zoom
        • Home - Reset camera
        
        Transform Tools:
        • G - Grab/Move
        • R - Rotate
        • S - Scale
        • X/Y/Z - Constrain to axis
        • Enter - Confirm
        • Esc - Cancel
        
        Modes:
        • Tab - Toggle Edit Mode
        • 1 - Vertex selection
        • 2 - Edge selection
        • 3 - Face selection
        
        Edit Operations:
        • E - Extrude (Face mode)
        • I - Inset (Face mode)
        • B - Bevel (Edge mode)
        • Ctrl+R - Subdivide
        • A - Select All
        
        General:
        • Shift+D - Duplicate
        • X or Delete - Delete
        • Ctrl+Z - Undo
        • Shift+Ctrl+Z - Redo
        • Ctrl+S - Save Scene
        • Ctrl+O - Load Scene
        • F12 - Screenshot
        """
        messagebox.showinfo("Keyboard Shortcuts", shortcuts)
    
    def show_about(self):
        about_text = """
        3D Modeling Engine v2.0
        
        An enhanced OpenGL-based 3D modeling application
        with professional tools and modern UI.
        
        Features:
        • Multiple primitive objects
        • Edit mode with vertex/edge/face selection
        • Transform tools (Grab, Rotate, Scale)
        • Modeling operations (Extrude, Inset, Bevel)
        • Material system with textures
        • Undo/Redo system
        • Scene save/load
        
        Built with Python, Pygame, and OpenGL
        """
        messagebox.showinfo("About", about_text)
    
    def update_hierarchy(self):
        self.object_listbox.delete(0, tk.END)
        display_objects = self.get_filtered_objects()
        
        for obj in display_objects:
            icon = "📦" if obj.object_type == "cube" else \
                   "⚪" if obj.object_type == "sphere" else \
                   "🔵" if obj.object_type == "cylinder" else \
                   "🔺" if obj.object_type == "cone" else \
                   "⭕" if obj.object_type == "torus" else \
                   "▭" if obj.object_type == "plane" else "🐵"
            
            display_name = f"{icon} {obj.name}"
            if obj.selected:
                display_name += " ✓"
            if not obj.visible:
                display_name += " 👁️"
            self.object_listbox.insert(tk.END, display_name)
    
    def update_properties(self):
        obj = self.engine.scene.get_selected_object()
        if obj:
            self.name_var.set(obj.name)
            self.pos_x_var.set(round(obj.position.x, 3))
            self.pos_y_var.set(round(obj.position.y, 3))
            self.pos_z_var.set(round(obj.position.z, 3))
            self.rot_x_var.set(round(obj.rotation.x, 3))
            self.rot_y_var.set(round(obj.rotation.y, 3))
            self.rot_z_var.set(round(obj.rotation.z, 3))
            self.scale_x_var.set(round(obj.scale.x, 3))
            self.scale_y_var.set(round(obj.scale.y, 3))
            self.scale_z_var.set(round(obj.scale.z, 3))
            
            info = f"Type: {obj.object_type.capitalize()}\n"
            info += f"Vertices: {len(obj.mesh.vertices)}\n"
            info += f"Edges: {len(obj.mesh.edges)}\n"
            info += f"Faces: {len(obj.mesh.faces)}"
            self.mesh_info_var.set(info)
            
            self.current_color = obj.material.color.copy()
            self.update_color_preview()
            self.shininess_var.set(obj.material.shininess)
            self.wireframe_var.set(obj.material.wireframe)
            self.shading_var.set(obj.material.shading_type)
            self.use_texture_var.set(obj.material.use_texture)
        else:
            self.name_var.set("")
            self.mesh_info_var.set("No object selected")
    
    def update_status(self):
        status = f"Mode: {self.engine.mode.upper()}"
        if self.engine.mode == "edit":
            status += f" ({self.engine.edit_mode_type.capitalize()})"
        status += f" | Objects: {len(self.engine.scene.objects)}"
        status += f" | Selected: {len(self.engine.scene.selected_objects)}"
        
        obj = self.engine.scene.get_selected_object()
        if obj:
            status += f" | {obj.name}"
            if self.engine.mode == "edit":
                status += f" | V:{len(obj.mesh.vertices)} F:{len(obj.mesh.faces)}"
        
        if self.engine.current_tool:
            status += f" | Tool: {self.engine.current_tool.upper()}"
            if self.engine.tool_axis:
                status += f" [{self.engine.tool_axis.upper()}]"
        
        self.status_var.set(status)
        
        # Update FPS
        try:
            fps = int(self.engine.clock.get_fps())
            self.fps_var.set(f"FPS: {fps}")
        except:
            pass
        
        self.draw_calls_var.set(f"Draw Calls: {len(self.engine.scene.objects)}")
    
    def start_update_loop(self):
        try:
            self.update_hierarchy()
            self.update_properties()
            self.update_status()
            self.update_timer = self.root.after(100, self.start_update_loop)
        except:
            pass
    
    def on_closing(self):
        if messagebox.askyesno("Quit", "Are you sure you want to exit?"):
            if self.update_timer:
                self.root.after_cancel(self.update_timer)
            self.engine.running = False

# ==================== Main Application ====================

class ModelingEngine:
    def __init__(self):
        pygame.init()
        
        self.width = 1200
        self.height = 900
        self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("3D Modeling Engine v2.0")
        
        self.background_color = (0.18, 0.18, 0.19)
        
        self.scene = SceneManager()
        self.camera = Camera()
        
        self.mode = "object"
        self.edit_mode_type = "vertex"
        
        self.mouse_down = False
        self.right_mouse_down = False
        self.middle_mouse_down = False
        self.last_mouse_pos = (0, 0)
        self.keys_pressed = set()
        
        self.show_grid = True
        self.show_axes = True
        
        self.current_tool = None
        self.tool_axis = None
        self.tool_start_mouse = None
        self.tool_start_values = {}
        
        self.history = []
        self.history_index = -1
        self.max_history = 50
        
        self.clock = pygame.time.Clock()
        self.running = True
        self.fps = 60
        
        self.setup_opengl()
        self.print_instructions()
        
        self.gui = None
        self.gui_ready = False
    
    def print_instructions(self):
        print("\n" + "="*70)
        print("ENHANCED 3D MODELING ENGINE v2.0")
        print("="*70)
        print("\nMODERN GUI: Use the Tkinter control panel for all operations")
        print("VIEWPORT: The main window is for 3D viewing and interaction")
        print("\nQUICK CONTROLS:")
        print("  Mouse: Rotate | Shift+Mouse: Pan | Wheel: Zoom")
        print("  G/R/S: Transform tools | Tab: Edit mode")
        print("  Press F1 in GUI for full keyboard shortcuts")
        print("="*70 + "\n")
    
    def setup_opengl(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        glLightfv(GL_LIGHT0, GL_POSITION, [10, 10, 5, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        glLightfv(GL_LIGHT0, GL_SPECULAR, [1, 1, 1, 1])
        
        glEnable(GL_NORMALIZE)
        glShadeModel(GL_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POINT_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
        
        glClearColor(*self.background_color, 1)
    
    def save_history_state(self):
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]
        
        state = {
            "objects": [obj.copy() for obj in self.scene.objects],
            "selected": [obj.name for obj in self.scene.selected_objects]
        }
        
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        else:
            self.history_index += 1
    
    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            state = self.history[self.history_index]
            self.restore_state(state)
            print("✓ Undo")
    
    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            state = self.history[self.history_index]
            self.restore_state(state)
            print("✓ Redo")
    
    def restore_state(self, state):
        self.scene.objects = [obj.copy() for obj in state["objects"]]
        self.scene.selected_objects.clear()
        
        for name in state["selected"]:
            for obj in self.scene.objects:
                if obj.name == name:
                    obj.selected = True
                    self.scene.selected_objects.append(obj)
                    break
    
    def start_transform_tool(self, tool_name):
        self.current_tool = tool_name
        self.tool_start_mouse = pygame.mouse.get_pos()
        self.tool_axis = None
        self.tool_start_values = {}
        
        if self.mode == "edit":
            obj = self.scene.get_selected_object()
            if obj:
                for i, vertex in enumerate(obj.mesh.vertices):
                    if vertex.selected:
                        self.tool_start_values[i] = vertex.position.copy()
        else:
            for obj in self.scene.selected_objects:
                if tool_name == "grab":
                    self.tool_start_values[obj] = obj.position.copy()
                elif tool_name == "rotate":
                    self.tool_start_values[obj] = obj.rotation.copy()
                elif tool_name == "scale":
                    self.tool_start_values[obj] = obj.scale.copy()
    
    def update_transform_tool(self):
        if not self.current_tool or not self.tool_start_mouse:
            return
        
        mouse_pos = pygame.mouse.get_pos()
        dx = (mouse_pos[0] - self.tool_start_mouse[0]) * 0.01
        dy = (mouse_pos[1] - self.tool_start_mouse[1]) * 0.01
        delta = dx - dy
        
        if self.mode == "edit":
            obj = self.scene.get_selected_object()
            if obj:
                for i, vertex in enumerate(obj.mesh.vertices):
                    if vertex.selected and i in self.tool_start_values:
                        start_pos = self.tool_start_values[i]
                        
                        if self.current_tool == "grab":
                            if self.tool_axis == 'x':
                                vertex.position.x = start_pos.x + delta * 2
                            elif self.tool_axis == 'y':
                                vertex.position.y = start_pos.y + delta * 2
                            elif self.tool_axis == 'z':
                                vertex.position.z = start_pos.z + delta * 2
                            else:
                                vertex.position.x = start_pos.x + dx * 2
                                vertex.position.z = start_pos.z - dy * 2
                        
                        elif self.current_tool == "scale":
                            scale_factor = 1.0 + delta
                            center = Vector3(0, 0, 0)
                            count = 0
                            for v in obj.mesh.vertices:
                                if v.selected:
                                    center = center + self.tool_start_values.get(obj.mesh.vertices.index(v), v.position)
                                    count += 1
                            if count > 0:
                                center = center / count
                            
                            direction = start_pos - center
                            vertex.position = center + direction * scale_factor
                
                obj.mesh.calculate_normals()
                obj.needs_update = True
        else:
            for obj in self.scene.selected_objects:
                start_value = self.tool_start_values.get(obj)
                if not start_value:
                    continue
                
                if self.current_tool == "grab":
                    if self.tool_axis == 'x':
                        obj.position.x = start_value.x + delta * 2
                    elif self.tool_axis == 'y':
                        obj.position.y = start_value.y + delta * 2
                    elif self.tool_axis == 'z':
                        obj.position.z = start_value.z + delta * 2
                    else:
                        obj.position.x = start_value.x + dx * 2
                        obj.position.z = start_value.z - dy * 2
                
                elif self.current_tool == "rotate":
                    angle = delta * 100
                    if self.tool_axis == 'x':
                        obj.rotation.x = start_value.x + angle
                    elif self.tool_axis == 'y':
                        obj.rotation.y = start_value.y + angle
                    elif self.tool_axis == 'z':
                        obj.rotation.z = start_value.z + angle
                    else:
                        obj.rotation.y = start_value.y + angle
                
                elif self.current_tool == "scale":
                    scale_factor = max(0.01, 1.0 + delta)
                    if self.tool_axis == 'x':
                        obj.scale.x = start_value.x * scale_factor
                    elif self.tool_axis == 'y':
                        obj.scale.y = start_value.y * scale_factor
                    elif self.tool_axis == 'z':
                        obj.scale.z = start_value.z * scale_factor
                    else:
                        obj.scale = start_value * scale_factor
                    
                    obj.scale.x = max(0.01, obj.scale.x)
                    obj.scale.y = max(0.01, obj.scale.y)
                    obj.scale.z = max(0.01, obj.scale.z)
    
    def confirm_transform_tool(self):
        if self.current_tool:
            self.save_history_state()
        self.current_tool = None
        self.tool_axis = None
        self.tool_start_mouse = None
        self.tool_start_values.clear()
    
    def cancel_transform_tool(self):
        if self.current_tool:
            if self.mode == "edit":
                obj = self.scene.get_selected_object()
                if obj:
                    for i, start_pos in self.tool_start_values.items():
                        if i < len(obj.mesh.vertices):
                            obj.mesh.vertices[i].position = start_pos.copy()
                    obj.mesh.calculate_normals()
                    obj.needs_update = True
            else:
                for obj in self.scene.selected_objects:
                    start_value = self.tool_start_values.get(obj)
                    if start_value:
                        if self.current_tool == "grab":
                            obj.position = start_value.copy()
                        elif self.current_tool == "rotate":
                            obj.rotation = start_value.copy()
                        elif self.current_tool == "scale":
                            obj.scale = start_value.copy()
        
        self.current_tool = None
        self.tool_axis = None
        self.tool_start_mouse = None
        self.tool_start_values.clear()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                self.running = False
            
            elif event.type == VIDEORESIZE:
                self.width, self.height = event.w, event.h
                self.screen = pygame.display.set_mode((self.width, self.height), DOUBLEBUF | OPENGL | RESIZABLE)
                glViewport(0, 0, self.width, self.height)
            
            elif event.type == KEYDOWN:
                self.keys_pressed.add(event.key)
                self.handle_keydown(event)
            
            elif event.type == KEYUP:
                self.keys_pressed.discard(event.key)
            
            elif event.type == MOUSEBUTTONDOWN:
                self.handle_mouse_down(event)
            
            elif event.type == MOUSEBUTTONUP:
                self.handle_mouse_up(event)
            
            elif event.type == MOUSEMOTION:
                self.handle_mouse_motion(event)
        
        if self.gui and self.gui_ready:
            try:
                self.gui.root.update()
            except:
                pass
    
    def handle_keydown(self, event):
        ctrl = pygame.key.get_mods() & KMOD_CTRL
        shift = pygame.key.get_mods() & KMOD_SHIFT
        
        if self.current_tool:
            if event.key == K_RETURN or event.key == K_KP_ENTER:
                self.confirm_transform_tool()
                return
            elif event.key == K_ESCAPE:
                self.cancel_transform_tool()
                return
            elif event.key == K_x:
                self.tool_axis = 'x'
                return
            elif event.key == K_y:
                self.tool_axis = 'y'
                return
            elif event.key == K_z and not ctrl:
                self.tool_axis = 'z'
                return
        
        if event.key == K_DELETE or (event.key == K_x and not ctrl and not self.current_tool):
            self.save_history_state()
            if self.mode == "edit":
                self.delete_selected_elements()
            else:
                self.scene.delete_selected()
        
        elif event.key == K_g:
            self.start_transform_tool("grab")
        elif event.key == K_r and not ctrl:
            self.start_transform_tool("rotate")
        elif event.key == K_s and not ctrl:
            self.start_transform_tool("scale")
        
        elif event.key == K_d and shift:
            self.save_history_state()
            self.scene.duplicate_selected()
        
        elif event.key == K_TAB:
            if self.mode == "object":
                if self.scene.get_selected_object():
                    self.mode = "edit"
            else:
                self.mode = "object"
        
        elif self.mode == "edit":
            if event.key == K_1:
                self.edit_mode_type = "vertex"
            elif event.key == K_2:
                self.edit_mode_type = "edge"
            elif event.key == K_3:
                self.edit_mode_type = "face"
            
            elif event.key == K_e:
                self.save_history_state()
                obj = self.scene.get_selected_object()
                if obj and self.edit_mode_type == "face":
                    obj.mesh.extrude_selected_faces(0.5)
                    obj.needs_update = True
            elif event.key == K_i:
                self.save_history_state()
                obj = self.scene.get_selected_object()
                if obj and self.edit_mode_type == "face":
                    obj.mesh.inset_selected_faces(0.2)
                    obj.needs_update = True
            elif event.key == K_b and not ctrl:
                self.save_history_state()
                obj = self.scene.get_selected_object()
                if obj and self.edit_mode_type == "edge":
                    obj.mesh.bevel_selected_edges(0.1)
                    obj.needs_update = True
            elif event.key == K_r and ctrl:
                self.save_history_state()
                obj = self.scene.get_selected_object()
                if obj:
                    obj.mesh.subdivide()
                    obj.needs_update = True
            elif event.key == K_a:
                self.toggle_select_all()
        
        elif event.key == K_s and ctrl:
            self.save_scene()
        elif event.key == K_o and ctrl:
            self.load_scene()
        
        elif event.key == K_z and ctrl:
            if shift:
                self.redo()
            else:
                self.undo()
        
        elif event.key == K_HOME:
            self.camera.reset()
        
        elif event.key == K_F12:
            self.take_screenshot()
        
        elif event.key == K_ESCAPE and not self.current_tool:
            self.running = False
    
    def delete_selected_elements(self):
        obj = self.scene.get_selected_object()
        if not obj:
            return
        
        if self.edit_mode_type == "vertex":
            new_vertices = []
            vertex_map = {}
            for i, v in enumerate(obj.mesh.vertices):
                if not v.selected:
                    vertex_map[i] = len(new_vertices)
                    new_vertices.append(v)
            
            obj.mesh.vertices = new_vertices
            
            new_faces = []
            for face in obj.mesh.faces:
                new_face_verts = []
                for v_idx in face.vertices:
                    if v_idx in vertex_map:
                        new_face_verts.append(vertex_map[v_idx])
                if len(new_face_verts) >= 3:
                    new_faces.append(Face(new_face_verts))
            
            obj.mesh.faces = new_faces
            obj.mesh.rebuild_edges()
        
        elif self.edit_mode_type == "face":
            obj.mesh.faces = [f for f in obj.mesh.faces if not f.selected]
            obj.mesh.rebuild_edges()
        
        obj.mesh.calculate_normals()
        obj.needs_update = True
    
    def toggle_select_all(self):
        obj = self.scene.get_selected_object()
        if not obj:
            return
        
        all_selected = True
        if self.edit_mode_type == "vertex":
            all_selected = all(v.selected for v in obj.mesh.vertices)
            for v in obj.mesh.vertices:
                v.selected = not all_selected
        elif self.edit_mode_type == "edge":
            all_selected = all(e.selected for e in obj.mesh.edges)
            for e in obj.mesh.edges:
                e.selected = not all_selected
        elif self.edit_mode_type == "face":
            all_selected = all(f.selected for f in obj.mesh.faces)
            for f in obj.mesh.faces:
                f.selected = not all_selected
    
    def save_scene(self):
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            ) if self.gui else "scene.json"
            
            if not filename and self.gui:
                return
            
            if not filename:
                filename = "scene.json"
            
            data = {"objects": [], "lights": []}
            
            for obj in self.scene.objects:
                obj_data = {
                    "name": obj.name,
                    "type": obj.object_type,
                    "position": obj.position.to_list(),
                    "rotation": obj.rotation.to_list(),
                    "scale": obj.scale.to_list(),
                    "material": {
                        "color": obj.material.color,
                        "ambient": obj.material.ambient,
                        "diffuse": obj.material.diffuse,
                        "specular": obj.material.specular,
                        "shininess": obj.material.shininess,
                        "wireframe": obj.material.wireframe,
                        "shading_type": obj.material.shading_type,
                        "use_texture": obj.material.use_texture,
                        "texture_path": obj.material.texture_path
                    },
                    "visible": obj.visible
                }
                data["objects"].append(obj_data)
            
            for light in self.scene.lights:
                light_data = {
                    "position": light.position.to_list(),
                    "ambient": light.ambient,
                    "diffuse": light.diffuse,
                    "specular": light.specular,
                    "enabled": light.enabled
                }
                data["lights"].append(light_data)
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Scene saved to {filename}")
            if self.gui:
                messagebox.showinfo("Success", f"Scene saved to {filename}")
        except Exception as e:
            print(f"✗ Error saving scene: {e}")
            if self.gui:
                messagebox.showerror("Error", f"Failed to save scene: {e}")
    
    def load_scene(self):
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            ) if self.gui else "scene.json"
            
            if not filename and self.gui:
                return
            
            if not filename:
                filename = "scene.json"
            
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.scene.objects.clear()
            self.scene.selected_objects.clear()
            
            for obj_data in data.get("objects", []):
                obj = Object3D(obj_data["name"], obj_data["type"])
                obj.generate_geometry()
                obj.position = Vector3(*obj_data["position"])
                obj.rotation = Vector3(*obj_data["rotation"])
                obj.scale = Vector3(*obj_data["scale"])
                
                mat = obj_data.get("material", {})
                obj.material.color = mat.get("color", [0.8, 0.8, 0.8])
                obj.material.ambient = mat.get("ambient", [0.2, 0.2, 0.2])
                obj.material.diffuse = mat.get("diffuse", [0.8, 0.8, 0.8])
                obj.material.specular = mat.get("specular", [1.0, 1.0, 1.0])
                obj.material.shininess = mat.get("shininess", 32.0)
                obj.material.wireframe = mat.get("wireframe", False)
                obj.material.shading_type = mat.get("shading_type", "smooth")
                obj.material.use_texture = mat.get("use_texture", False)
                
                if mat.get("texture_path") and mat["texture_path"].startswith("procedural_"):
                    pattern = mat["texture_path"].replace("procedural_", "")
                    obj.material.create_procedural_texture(pattern)
                
                obj.visible = obj_data.get("visible", True)
                self.scene.objects.append(obj)
            
            print(f"✓ Loaded {len(self.scene.objects)} objects from {filename}")
            if self.gui:
                messagebox.showinfo("Success", f"Loaded {len(self.scene.objects)} objects")
        except FileNotFoundError:
            print("⚠ No scene file found")
            if self.gui:
                messagebox.showwarning("Warning", "Scene file not found")
        except Exception as e:
            print(f"✗ Error loading scene: {e}")
            if self.gui:
                messagebox.showerror("Error", f"Failed to load scene: {e}")
    
    def take_screenshot(self):
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            
            x, y, width, height = glGetIntegerv(GL_VIEWPORT)
            glReadBuffer(GL_FRONT)
            pixels = glReadPixels(x, y, width, height, GL_RGB, GL_UNSIGNED_BYTE)
            
            image = Image.frombytes("RGB", (width, height), pixels)
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            image.save(filename)
            
            print(f"✓ Screenshot saved as {filename}")
            if self.gui:
                messagebox.showinfo("Success", f"Screenshot saved as {filename}")
        except Exception as e:
            print(f"✗ Error taking screenshot: {e}")
            if self.gui:
                messagebox.showerror("Error", f"Failed to save screenshot: {e}")
    
    def handle_mouse_down(self, event):
        shift = pygame.key.get_mods() & KMOD_SHIFT
        
        if event.button == 1:
            if self.mode == "edit":
                self.select_edit_element(event.pos)
            else:
                self.mouse_down = True
            self.last_mouse_pos = event.pos
        elif event.button == 2:
            self.middle_mouse_down = True
            self.last_mouse_pos = event.pos
        elif event.button == 3:
            if shift:
                self.middle_mouse_down = True
            else:
                self.right_mouse_down = True
            self.last_mouse_pos = event.pos
        elif event.button == 4:
            self.camera.zoom(-1)
        elif event.button == 5:
            self.camera.zoom(1)
    
    def select_edit_element(self, mouse_pos):
        obj = self.scene.get_selected_object()
        if not obj:
            return
        
        if self.edit_mode_type == "vertex":
            min_dist = float('inf')
            closest_vertex = None
            
            for vertex in obj.mesh.vertices:
                screen_pos = self.project_to_screen(vertex.position, obj)
                if screen_pos:
                    dist = math.sqrt((screen_pos[0] - mouse_pos[0])**2 + 
                                   (screen_pos[1] - mouse_pos[1])**2)
                    if dist < min_dist and dist < 20:
                        min_dist = dist
                        closest_vertex = vertex
            
            if closest_vertex:
                shift = pygame.key.get_mods() & KMOD_SHIFT
                if not shift:
                    for v in obj.mesh.vertices:
                        v.selected = False
                closest_vertex.selected = not closest_vertex.selected
        
        elif self.edit_mode_type == "edge":
            min_dist = float('inf')
            closest_edge = None
            
            for edge in obj.mesh.edges:
                if edge.v1 < len(obj.mesh.vertices) and edge.v2 < len(obj.mesh.vertices):
                    v1_screen = self.project_to_screen(obj.mesh.vertices[edge.v1].position, obj)
                    v2_screen = self.project_to_screen(obj.mesh.vertices[edge.v2].position, obj)
                    
                    if v1_screen and v2_screen:
                        dist = self.point_to_line_distance(mouse_pos, v1_screen, v2_screen)
                        if dist < min_dist and dist < 15:
                            min_dist = dist
                            closest_edge = edge
            
            if closest_edge:
                shift = pygame.key.get_mods() & KMOD_SHIFT
                if not shift:
                    for e in obj.mesh.edges:
                        e.selected = False
                closest_edge.selected = not closest_edge.selected
        
        elif self.edit_mode_type == "face":
            min_dist = float('inf')
            closest_face = None
            
            for face in obj.mesh.faces:
                center_3d = face.get_center(obj.mesh.vertices)
                center_screen = self.project_to_screen(center_3d, obj)
                
                if center_screen:
                    dist = math.sqrt((center_screen[0] - mouse_pos[0])**2 + 
                                   (center_screen[1] - mouse_pos[1])**2)
                    if dist < min_dist and dist < 30:
                        min_dist = dist
                        closest_face = face
            
            if closest_face:
                shift = pygame.key.get_mods() & KMOD_SHIFT
                if not shift:
                    for f in obj.mesh.faces:
                        f.selected = False
                closest_face.selected = not closest_face.selected
    
    def point_to_line_distance(self, point, line_start, line_end):
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end
        
        line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
        
        if line_len_sq == 0:
            return math.sqrt((px - x1)**2 + (py - y1)**2)
        
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
        
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        
        return math.sqrt((px - closest_x)**2 + (py - closest_y)**2)
    
    def project_to_screen(self, pos, obj):
        try:
            world_pos = pos.copy()
            world_pos.x *= obj.scale.x
            world_pos.y *= obj.scale.y
            world_pos.z *= obj.scale.z
            world_pos = world_pos + obj.position
            
            modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
            projection = glGetDoublev(GL_PROJECTION_MATRIX)
            viewport = glGetIntegerv(GL_VIEWPORT)
            
            screen_pos = gluProject(world_pos.x, world_pos.y, world_pos.z,
                                   modelview, projection, viewport)
            
            return (screen_pos[0], self.height - screen_pos[1])
        except:
            return None
    
    def handle_mouse_up(self, event):
        if event.button == 1:
            self.mouse_down = False
        elif event.button == 2:
            self.middle_mouse_down = False
        elif event.button == 3:
            self.right_mouse_down = False
    
    def handle_mouse_motion(self, event):
        if self.current_tool:
            self.update_transform_tool()
            return
        
        x, y = event.pos
        dx = x - self.last_mouse_pos[0]
        dy = y - self.last_mouse_pos[1]
        
        shift = pygame.key.get_mods() & KMOD_SHIFT
        
        if self.mouse_down and not shift:
            self.camera.rotate(dx, dy)
        elif self.middle_mouse_down or (self.mouse_down and shift):
            self.camera.pan(-dx, dy)
        elif self.right_mouse_down and not shift:
            self.camera.zoom(dy * 0.1)
        
        self.last_mouse_pos = (x, y)
    
    def add_object(self, object_type: str):
        self.save_history_state()
        obj = self.scene.add_object(object_type)
        self.scene.clear_selection()
        self.scene.select_object(obj)
        print(f"✓ Added {object_type}")
    
    def draw_grid(self):
        if not self.show_grid:
            return
        
        glDisable(GL_LIGHTING)
        glBegin(GL_LINES)
        for i in range(-20, 21):
            if i == 0:
                glColor3f(0.5, 0.5, 0.5)
            else:
                glColor3f(0.3, 0.3, 0.3)
            
            glVertex3f(-20, 0, i)
            glVertex3f(20, 0, i)
            glVertex3f(i, 0, -20)
            glVertex3f(i, 0, 20)
        glEnd()
        glEnable(GL_LIGHTING)
    
    def draw_axes(self):
        if not self.show_axes:
            return
        
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(5, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 5, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 5)
        glEnd()
        glLineWidth(1.0)
        glEnable(GL_LIGHTING)
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.camera.apply(self.width, self.height)
        
        for light in self.scene.lights:
            light.apply()
        
        self.draw_grid()
        self.draw_axes()
        
        for obj in self.scene.objects:
            obj.draw(self.mode)
        
        pygame.display.flip()
    
    def run(self):
        self.gui = TkinterGUI(self)
        self.gui_ready = True
        
        while self.running:
            self.handle_events()
            self.render()
            self.clock.tick(self.fps)
        
        if self.gui:
            try:
                self.gui.root.destroy()
            except:
                pass
        pygame.quit()
        print("\n✓ Thank you for using 3D Modeling Engine v2.0!")

# ==================== Entry Point ====================

def main():
    try:
        app = ModelingEngine()
        app.run()
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()