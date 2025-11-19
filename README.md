# PixelForge3D v2.0

**A full-featured 3D modeler written entirely in Python**  
Blender-inspired workflow · Object & Edit Mode · Real-time OpenGL rendering · Modern dark UI

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![OpenGL](https://img.shields.io/badge/OpenGL-3.3-red)
![License](https://img.shields.io/badge/License-MIT-green)
[![Stars](https://img.shields.io/github/stars/Zypher0903/PixelForge3D?style=social)](https://github.com/Zypher0903/PixelForge3D/stargazers)

https://github.com/user-attachments/assets/your-future-gif-here.gif

> **Warning** Still adding screenshots & demo GIFs – they’re coming in the next 24h!

### Features

| Feature                        | Status |
|-------------------------------|--------|
| Full **Object Mode** & **Edit Mode** (Vertex/Edge/Face) | Done |
| Modeling tools: **Extrude · Inset · Bevel · Subdivide** | Done |
| Transform tools (G/R/S) with axis constraints (X/Y/Z) | Done |
| Material editor + procedural textures (checker, grid, gradient) | Done |
| Unlimited **Undo/Redo** (Ctrl+Z / Ctrl+Shift+Z) | Done |
| Save/Load scenes (.json) | Done |
| Modern dark Tkinter GUI with tabs & icons | Done |
| Real-time FPS counter & status bar | Done |
| Screenshot capture (F12) | Done |
| Wireframe · Smooth/Flat shading · Texture toggle | Done |

### Blender-style Hotkeys

| Key              | Action                    |
|------------------|---------------------------|
| `G` / `R` / `S`  | Grab / Rotate / Scale     |
| `X` `Y` `Z`      | Constrain to axis         |
| `Tab`            | Toggle Object ↔ Edit Mode |
| `1` `2` `3`      | Vertex · Edge · Face mode |
| `E`              | Extrude (Face)            |
| `I`              | Inset (Face)              |
| `B`              | Bevel (Edge)              |
| `Ctrl+R`         | Subdivide                 |
| `A`              | Select All / Deselect     |
| `Shift+D`        | Duplicate                 |
| `Delete` / `X`   | Delete                    |
| `Ctrl+Z`         | Undo                      |
| `Ctrl+Shift+Z`   | Redo                      |
| `Ctrl+S` / `Ctrl+O` | Save / Load scene      |
| `F12`            | Screenshot                |
| `Home`           | Reset camera              |

### Quick Start (30 seconds)

```bash
git clone https://github.com/Zypher0903/PixelForge3D.git
cd PixelForge3D
pip install -r requirements.txt
python main.py
Requirements
txtpygame==2.6.0
PyOpenGL==3.1.7
PyOpenGL_accelerate
numpy
Pillow
(Everything is in requirements.txt)
Screenshots (coming today/tomorrow)
Main View
Edit Mode
Material Tab
Extrude Demo
Why Python + OpenGL?

Zero dependencies beyond Python
Runs on Windows, macOS, Linux
Perfect for learning 3D graphics, mesh editing, and GUI design
Easily extensible (add modifiers, import/export, shaders, etc.)

Roadmap (what's cooking)

 Proper object picking in viewport (color picking / raycasting)
 .OBJ export/import
 Mirror & Array modifiers
 Image texture loading
 VBO rendering for 100k+ polygons
 Toolbars with icons
 Release v3.0 with PySide6/Qt version (optional)

Author
Zypher0903 · 2025
Made with passion, coffee, and way too many late nights
If you like this project — drop a star, it means the world!

PixelForge3D — Forging pixels into 3D masterpieces, one Python line at a time.
