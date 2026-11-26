"""Blender (Cycles) renderer for one worksheet object.

Usage:
  blender -b --factory-startup -P blender_render.py -- spec.json out_dir [samples]

Writes into out_dir:
  photo.png   - photoreal studio product shot (Cycles)
  labels.png  - flat per-part label render (Workbench, no AA, raw colours)
  geo.json    - projected construction ellipses + centre axis (pixel coords)
"""
import bpy, bmesh, json, math, sys, os
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv[sys.argv.index("--") + 1:]
SPEC = json.load(open(argv[0]))
OUT = argv[1]
SAMPLES = int(argv[2]) if len(argv) > 2 else 160
RES = int(SPEC.get("res", 950))
if len(argv) > 3:
    RES = int(argv[3])
os.makedirs(OUT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ---------------------------------------------------------------- materials
def _nodes(mat):
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    return nt


def _principled(nt):
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    p = nt.nodes.new("ShaderNodeBsdfPrincipled")
    nt.links.new(p.outputs["BSDF"], out.inputs["Surface"])
    return p, out


def _coord(nt):
    tc = nt.nodes.new("ShaderNodeTexCoord")
    return tc.outputs["Object"]


def _noise(nt, scale, detail=4.0, rough=0.55, distortion=0.0, vec=None):
    n = nt.nodes.new("ShaderNodeTexNoise")
    n.inputs["Scale"].default_value = scale
    n.inputs["Detail"].default_value = detail
    n.inputs["Roughness"].default_value = rough
    n.inputs["Distortion"].default_value = distortion
    if vec is not None:
        nt.links.new(vec, n.inputs["Vector"])
    return n


def _ramp(nt, fac, stops):
    r = nt.nodes.new("ShaderNodeValToRGB")
    els = r.color_ramp.elements
    while len(els) > 2:
        els.remove(els[-1])
    for i, (pos, col) in enumerate(stops):
        e = els[i] if i < 2 else els.new(pos)
        e.position = pos
        e.color = (*col[:3], 1.0)
    nt.links.new(fac, r.inputs["Fac"])
    return r


def _math(nt, op, a, b=None, val=None):
    m = nt.nodes.new("ShaderNodeMath")
    m.operation = op
    if isinstance(a, (int, float)):
        m.inputs[0].default_value = a
    else:
        nt.links.new(a, m.inputs[0])
    if b is not None:
        if isinstance(b, (int, float)):
            m.inputs[1].default_value = b
        else:
            nt.links.new(b, m.inputs[1])
    return m.outputs[0]


def _bump(nt, height, strength, distance, normal=None):
    b = nt.nodes.new("ShaderNodeBump")
    b.inputs["Strength"].default_value = strength
    b.inputs["Distance"].default_value = distance
    nt.links.new(height, b.inputs["Height"])
    if normal is not None:
        nt.links.new(normal, b.inputs["Normal"])
    return b.outputs["Normal"]


def _zfac(nt, vec, z0, z1):
    """0..1 factor along object Z between z0 and z1."""
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(vec, sep.inputs[0])
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = z0
    mr.inputs["From Max"].default_value = z1
    nt.links.new(sep.outputs["Z"], mr.inputs["Value"])
    return mr.outputs["Result"]


def make_material(name, m):
    mat = bpy.data.materials.new(name)
    nt = _nodes(mat)
    p, out = _principled(nt)
    kind = m.get("kind", "ceramic")
    col = m.get("color", [0.8, 0.8, 0.8])
    vec = _coord(nt)
    H = SPEC["height"]

    def base_color_with_variation(c1, c2=None, var=0.06, scale=4.0, z0=0.0, z1=None):
        """colour = gradient(c2 at bottom -> c1 at top) modulated by soft noise"""
        n = _noise(nt, scale / H, 3.0, 0.5, vec=vec)
        if c2 is not None:
            zf = _zfac(nt, vec, z0 if z0 else 0.0, z1 if z1 else H)
            mix = _math(nt, "ADD", zf, _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", n.outputs["Fac"], 0.5), 0.35))
            r = _ramp(nt, mix, [(0.0, c2), (1.0, c1)])
        else:
            lo = [max(0.0, c * (1 - var)) for c in c1]
            hi = [min(1.0, c * (1 + var)) for c in c1]
            r = _ramp(nt, n.outputs["Fac"], [(0.3, lo), (0.7, hi)])
        return r.outputs["Color"]

    if kind in ("ceramic", "porcelain"):
        c = base_color_with_variation(col, m.get("color2"), var=m.get("var", 0.05),
                                      z0=m.get("grad_z0", 0.0), z1=m.get("grad_z1"))
        if m.get("speckle"):
            v = nt.nodes.new("ShaderNodeTexVoronoi")
            v.inputs["Scale"].default_value = 260.0 * 0.3 / H
            nt.links.new(vec, v.inputs["Vector"])
            sp = _ramp(nt, v.outputs["Distance"], [(0.0, [0.12, 0.1, 0.08]), (0.07, [1, 1, 1])])
            mul = nt.nodes.new("ShaderNodeMix")
            mul.data_type = "RGBA"
            mul.blend_type = "MULTIPLY"
            mul.inputs[0].default_value = m.get("speckle", 0.8)
            nt.links.new(c, mul.inputs[6])
            nt.links.new(sp.outputs["Color"], mul.inputs[7])
            c = mul.outputs[2]
        nt.links.new(c, p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = m.get("roughness", 0.18)
        p.inputs["Coat Weight"].default_value = m.get("coat", 0.6)
        p.inputs["Coat Roughness"].default_value = m.get("coat_rough", 0.06)
        p.inputs["Specular IOR Level"].default_value = 0.5
        n = _noise(nt, 9.0 / H, 6.0, 0.6, vec=vec)
        nt.links.new(_bump(nt, n.outputs["Fac"], m.get("bump", 0.06), 0.002), p.inputs["Normal"])
        if m.get("subsurface"):
            p.inputs["Subsurface Weight"].default_value = m["subsurface"]
            p.inputs["Subsurface Radius"].default_value = (0.4, 0.3, 0.2)
            p.inputs["Subsurface Scale"].default_value = 0.004
    elif kind in ("matte", "terracotta", "stone", "concrete"):
        c = base_color_with_variation(col, m.get("color2"), var=m.get("var", 0.10), scale=6.0)
        nt.links.new(c, p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = m.get("roughness", 0.78)
        p.inputs["Specular IOR Level"].default_value = 0.3
        n = _noise(nt, 60.0 * 0.3 / H, 8.0, 0.7, vec=vec)
        nt.links.new(_bump(nt, n.outputs["Fac"], m.get("bump", 0.25), 0.002), p.inputs["Normal"])
    elif kind == "glass":
        # let shadow rays mostly pass so things inside glass get lit (caustics are off), keep a faint shadow
        lp = nt.nodes.new("ShaderNodeLightPath")
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        tr.inputs["Color"].default_value = (*[0.55 + 0.45 * c for c in col], 1)
        mx = nt.nodes.new("ShaderNodeMixShader")
        fac = _math(nt, "MULTIPLY", lp.outputs["Is Shadow Ray"], 0.92)
        nt.links.new(fac, mx.inputs[0])
        nt.links.new(p.outputs["BSDF"], mx.inputs[1])
        nt.links.new(tr.outputs[0], mx.inputs[2])
        nt.links.new(mx.outputs[0], out.inputs["Surface"])
        p.inputs["Base Color"].default_value = (*col, 1)
        p.inputs["Transmission Weight"].default_value = 1.0
        p.inputs["Roughness"].default_value = m.get("roughness", 0.0)
        p.inputs["IOR"].default_value = m.get("ior", 1.48)
        p.inputs["Specular IOR Level"].default_value = 0.5
        n = _noise(nt, 3.0 / H, 2.0, 0.4, vec=vec)
        nt.links.new(_bump(nt, n.outputs["Fac"], 0.02, 0.003), p.inputs["Normal"])
    elif kind == "metal":
        nvar = _noise(nt, 5.0 / H, 4.0, 0.6, vec=vec)
        lo = [c * 0.82 for c in col]
        r = _ramp(nt, nvar.outputs["Fac"], [(0.25, lo), (0.75, col)])
        nt.links.new(r.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Metallic"].default_value = 1.0
        rough = m.get("roughness", 0.22)
        rn = _noise(nt, 14.0 / H, 6.0, 0.6, vec=vec)
        rr = _ramp(nt, rn.outputs["Fac"], [(0.2, [rough * 0.75] * 3), (0.8, [rough * 1.3] * 3)])
        nt.links.new(rr.outputs["Color"], p.inputs["Roughness"])
        if m.get("brushed"):
            p.inputs["Anisotropic"].default_value = 0.7
        n = _noise(nt, 30.0 / H, 6.0, 0.6, vec=vec)
        nt.links.new(_bump(nt, n.outputs["Fac"], m.get("bump", 0.03), 0.001), p.inputs["Normal"])
    elif kind == "fabric":
        # linen/cotton lampshade: weave bump, sheen, a bit of translucency
        c = base_color_with_variation(col, None, var=0.05, scale=12.0)
        nt.links.new(c, p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = 0.95
        p.inputs["Specular IOR Level"].default_value = 0.15
        p.inputs["Sheen Weight"].default_value = 0.4
        p.inputs["Subsurface Weight"].default_value = 0.15
        p.inputs["Subsurface Scale"].default_value = 0.003
        w1 = nt.nodes.new("ShaderNodeTexWave")
        w1.wave_type = "BANDS"
        w1.bands_direction = "Z"
        w1.inputs["Scale"].default_value = 900 * 0.3 / H
        w1.inputs["Distortion"].default_value = 2.0
        nt.links.new(vec, w1.inputs["Vector"])
        n = _noise(nt, 300.0 * 0.3 / H, 4.0, 0.6, vec=vec)
        h = _math(nt, "ADD", _math(nt, "MULTIPLY", w1.outputs["Fac"], 0.5), n.outputs["Fac"])
        nt.links.new(_bump(nt, h, 0.25, 0.0008), p.inputs["Normal"])
    elif kind == "wood":
        w = nt.nodes.new("ShaderNodeTexWave")
        w.wave_type = "BANDS"
        w.bands_direction = "Z"
        w.inputs["Scale"].default_value = m.get("grain", 14.0) * 0.3 / H
        w.inputs["Distortion"].default_value = 6.0
        w.inputs["Detail"].default_value = 3.0
        w.inputs["Detail Scale"].default_value = 1.5
        mp = nt.nodes.new("ShaderNodeMapping")
        mp.inputs["Scale"].default_value = (0.15, 0.15, 1.0)
        nt.links.new(vec, mp.inputs["Vector"])
        # grain runs vertically: rotate so bands follow X
        mp.inputs["Rotation"].default_value = (0, math.radians(90), 0)
        nt.links.new(mp.outputs["Vector"], w.inputs["Vector"])
        c2 = m.get("color2", [c * 0.55 for c in col])
        r = _ramp(nt, w.outputs["Fac"], [(0.15, c2), (0.85, col)])
        nt.links.new(r.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = m.get("roughness", 0.45)
        p.inputs["Coat Weight"].default_value = m.get("coat", 0.25)
        nt.links.new(_bump(nt, w.outputs["Fac"], 0.08, 0.001), p.inputs["Normal"])
    elif kind == "marble":
        n = _noise(nt, 3.0 / H, 10.0, 0.62, distortion=4.0, vec=vec)
        r = _ramp(nt, n.outputs["Fac"], [(0.45, col), (0.5, m.get("vein", [0.35, 0.35, 0.36])), (0.55, col)])
        nt.links.new(r.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = m.get("roughness", 0.12)
        p.inputs["Coat Weight"].default_value = 0.3
    elif kind == "wax":
        p.inputs["Base Color"].default_value = (*col, 1)
        p.inputs["Roughness"].default_value = 0.45
        p.inputs["Subsurface Weight"].default_value = 0.6
        p.inputs["Subsurface Radius"].default_value = (1.0, 0.6, 0.35)
        p.inputs["Subsurface Scale"].default_value = 0.01
    elif kind == "plastic":
        p.inputs["Base Color"].default_value = (*col, 1)
        p.inputs["Roughness"].default_value = m.get("roughness", 0.3)
    elif kind == "banded":
        # horizontal paint bands: bands = [[z_from_m, [r,g,b]], ...] sorted by z
        zf = _zfac(nt, vec, 0.0, H)
        rr = nt.nodes.new("ShaderNodeValToRGB")
        rr.color_ramp.interpolation = "CONSTANT"
        els = rr.color_ramp.elements
        bands = m["bands"]
        els[0].position = 0.0
        els[0].color = (*bands[0][1], 1)
        els[1].position = min(0.999, max(0.001, bands[1][0] / H)) if len(bands) > 1 else 0.999
        els[1].color = (*(bands[1][1] if len(bands) > 1 else bands[0][1]), 1)
        for zb, cb in bands[2:]:
            e = els.new(min(0.999, max(0.001, zb / H)))
            e.color = (*cb, 1)
        nt.links.new(zf, rr.inputs["Fac"])
        nt.links.new(rr.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = m.get("roughness", 0.35)
        p.inputs["Metallic"].default_value = m.get("metallic", 0.0)
        p.inputs["Coat Weight"].default_value = m.get("coat", 0.0)
        n = _noise(nt, 20.0 / H, 4.0, 0.6, vec=vec)
        nt.links.new(_bump(nt, n.outputs["Fac"], m.get("bump", 0.05), 0.001), p.inputs["Normal"])
    elif kind == "segments":
        # vertical segments around the axis (beach ball, watermelon stripes)
        sep = nt.nodes.new("ShaderNodeSeparateXYZ")
        nt.links.new(vec, sep.inputs[0])
        at = _math(nt, "ARCTAN2", sep.outputs["Y"], sep.outputs["X"])
        a01 = _math(nt, "ADD", _math(nt, "DIVIDE", at, 2 * math.pi), 0.5)
        if m.get("wobble"):
            nn = _noise(nt, m.get("wobble_scale", 6.0) / H, 3.0, 0.6, vec=vec)
            a01 = _math(nt, "ADD", a01, _math(nt, "MULTIPLY", _math(nt, "SUBTRACT", nn.outputs["Fac"], 0.5),
                                              m["wobble"]))
        a01 = _math(nt, "FRACT", a01)
        cols = m["colors"]
        nseg = m.get("n", len(cols))
        rr = nt.nodes.new("ShaderNodeValToRGB")
        rr.color_ramp.interpolation = "CONSTANT"
        els = rr.color_ramp.elements
        els[0].position = 0.0
        els[0].color = (*cols[0], 1)
        els[1].position = 1.0 / nseg
        els[1].color = (*cols[1 % len(cols)], 1)
        for k in range(2, nseg):
            e = els.new(k / nseg)
            e.color = (*cols[k % len(cols)], 1)
        nt.links.new(a01, rr.inputs["Fac"])
        nt.links.new(rr.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = m.get("roughness", 0.3)
        p.inputs["Coat Weight"].default_value = m.get("coat", 0.0)
        n = _noise(nt, 30.0 / H, 4.0, 0.6, vec=vec)
        nt.links.new(_bump(nt, n.outputs["Fac"], m.get("bump", 0.04), 0.001), p.inputs["Normal"])
    elif kind == "globe":
        n = _noise(nt, m.get("scale", 2.2) / H, 6.0, 0.62, distortion=0.6, vec=vec)
        rr = _ramp(nt, n.outputs["Fac"], [(0.0, [0.03, 0.12, 0.30]), (0.53, [0.06, 0.22, 0.42])])
        e1 = rr.color_ramp.elements.new(0.535)
        e1.color = (0.52, 0.48, 0.30, 1)
        e2 = rr.color_ramp.elements.new(0.62)
        e2.color = (0.25, 0.38, 0.14, 1)
        e3 = rr.color_ramp.elements.new(0.72)
        e3.color = (0.45, 0.33, 0.18, 1)
        nt.links.new(rr.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = 0.35
        p.inputs["Coat Weight"].default_value = 0.6
    elif kind == "glow":
        p.inputs["Base Color"].default_value = (*col, 1)
        p.inputs["Roughness"].default_value = 0.3
        p.inputs["Emission Color"].default_value = (*col, 1)
        p.inputs["Emission Strength"].default_value = m.get("strength", 1.5)
    elif kind == "sand":
        n = _noise(nt, 900.0 * 0.3 / H, 2.0, 0.5, vec=vec)
        r = _ramp(nt, n.outputs["Fac"], [(0.3, [c * 0.8 for c in col]), (0.7, col)])
        nt.links.new(r.outputs["Color"], p.inputs["Base Color"])
        p.inputs["Roughness"].default_value = 0.95
        nt.links.new(_bump(nt, n.outputs["Fac"], 0.5, 0.001), p.inputs["Normal"])
    elif kind == "emission_bulb":
        p.inputs["Base Color"].default_value = (*col, 1)
        p.inputs["Roughness"].default_value = 0.35
        p.inputs["Transmission Weight"].default_value = 0.4
    else:
        p.inputs["Base Color"].default_value = (*col, 1)
    return mat


# ---------------------------------------------------------------- geometry
def _finish(name, verts, faces, material, smooth=True):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-7)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    ob = bpy.data.objects.new(name, me)
    scene.collection.objects.link(ob)
    ob.data.materials.append(material)
    return ob


def lathe(name, prof, material, segs=160, loop=False, offset=(0, 0, 0), deform=None, rot=0.0):
    """deform: optional dict
         spout: {z0, z1, amp, dir_deg, power}  pulls the rim out into a pouring lip
         flutes: {n, amp, z0, z1, sharp}       vertical ribs / lobes (radius modulation)"""
    verts, faces, rings = [], [], []
    dfm = deform or {}
    sp = dfm.get("spout")
    fl = dfm.get("flutes")
    for (r, z) in prof:
        if r < 1e-6:
            rings.append([len(verts)])
            verts.append((offset[0], offset[1], offset[2] + z))
        else:
            ids = []
            for j in range(segs):
                a = 2 * math.pi * j / segs + rot
                rr = r
                if fl and fl["z0"] <= z <= fl["z1"]:
                    w = 1.0
                    ez = fl.get("ease", 0.0)
                    if ez > 0:
                        w = min(1.0, (z - fl["z0"]) / ez, (fl["z1"] - z) / ez)
                        w = w * w * (3 - 2 * w)
                    c = 0.5 + 0.5 * math.cos(fl["n"] * a)
                    c = c ** fl.get("sharp", 1.0)
                    rr = rr * (1 + w * fl["amp"] * (c - 0.5))
                if sp and z > sp["z0"]:
                    t = min(1.0, (z - sp["z0"]) / max(1e-6, sp["z1"] - sp["z0"]))
                    da = a - math.radians(sp.get("dir_deg", 180))
                    c = max(0.0, math.cos(da)) ** sp.get("power", 8)
                    rr = rr + sp["amp"] * (t ** 2) * c
                ids.append(len(verts))
                verts.append((offset[0] + rr * math.cos(a), offset[1] + rr * math.sin(a), offset[2] + z))
            rings.append(ids)
    n = len(rings)
    pairs = [(i, i + 1) for i in range(n - 1)]
    if loop:
        pairs.append((n - 1, 0))
    for i, k in pairs:
        A, B = rings[i], rings[k]
        if len(A) == 1 and len(B) == 1:
            continue
        if len(A) == 1:
            for j in range(segs):
                faces.append((A[0], B[(j + 1) % segs], B[j]))
        elif len(B) == 1:
            for j in range(segs):
                faces.append((A[j], A[(j + 1) % segs], B[0]))
        else:
            for j in range(segs):
                faces.append((A[j], A[(j + 1) % segs], B[(j + 1) % segs], B[j]))
    return _finish(name, verts, faces, material)


def sweep(name, path, radii, material, segs=40, caps=True, up=(0, 1, 0)):
    """Tube along 3D path; radii = [(a,b)] ellipse half-axes (a in-plane, b along 'up')."""
    P = [Vector(p) for p in path]
    n = len(P)
    T = []
    for i in range(n):
        d = (P[min(i + 1, n - 1)] - P[max(i - 1, 0)])
        T.append(d.normalized())
    B0 = Vector(up)
    verts, faces, rings = [], [], []
    Bp = (B0 - B0.dot(T[0]) * T[0]).normalized()
    for i in range(n):
        if i > 0:  # parallel transport
            Bp = (Bp - Bp.dot(T[i]) * T[i]).normalized()
        N = T[i].cross(Bp).normalized()
        a, b = radii[i]
        ids = []
        for j in range(segs):
            t = 2 * math.pi * j / segs
            v = P[i] + N * (a * math.cos(t)) + Bp * (b * math.sin(t))
            ids.append(len(verts))
            verts.append(tuple(v))
        rings.append(ids)
    for i in range(n - 1):
        A, Bq = rings[i], rings[i + 1]
        for j in range(segs):
            faces.append((A[j], A[(j + 1) % segs], Bq[(j + 1) % segs], Bq[j]))
    if caps:
        for idx, ring in ((0, rings[0]), (n - 1, rings[-1])):
            c = len(verts)
            verts.append(tuple(P[idx]))
            for j in range(segs):
                faces.append((ring[j], ring[(j + 1) % segs], c))
    return _finish(name, verts, faces, material)


def box(name, center, size, material, bevel=0.002):
    bpy.ops.mesh.primitive_cube_add(size=1, location=center)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = size
    bpy.ops.object.transform_apply(scale=True)
    md = ob.modifiers.new("bev", "BEVEL")
    md.width = bevel
    md.segments = 3
    ob.data.materials.append(material)
    for pg in ob.data.polygons:
        pg.use_smooth = False
    return ob


# ---------------------------------------------------------------- build object
mats = {k: make_material(k, v) for k, v in SPEC["materials"].items()}
parts = []
for i, pt in enumerate(SPEC["parts"]):
    nm = f"part{i:02d}"
    mat = mats[pt["material"]]
    if pt["type"] == "lathe":
        ob = lathe(nm, pt["profile"], mat, segs=pt.get("segs", 160), loop=pt.get("loop", False),
                   offset=pt.get("offset", (0, 0, 0)), deform=pt.get("deform"), rot=pt.get("rot", 0.0))
        if pt.get("flat"):
            for pg in ob.data.polygons:
                pg.use_smooth = False
    elif pt["type"] == "sweep":
        ob = sweep(nm, pt["path"], pt["radii"], mat, segs=pt.get("segs", 40), caps=pt.get("caps", True),
                   up=pt.get("up", (0, 1, 0)))
    elif pt["type"] == "box":
        ob = box(nm, pt["center"], pt["size"], mat, pt.get("bevel", 0.002))
    else:
        raise ValueError(pt["type"])
    if pt.get("rot_z"):
        ob.rotation_euler[2] = math.radians(pt["rot_z"])
    if pt.get("rot3"):
        # rotate about the part's own local origin, then move it into place
        ob.rotation_euler = [math.radians(a) for a in pt["rot3"]]
        if pt.get("loc3"):
            ob.location = Vector(pt["loc3"])
    lab = int(pt.get("label", (i % 30) + 1))
    ob["label"] = ((lab - 1) % 31) + 1
    parts.append(ob)

H = SPEC["height"]
W = SPEC["width"]       # max horizontal extent (x) of the object
Dp = SPEC.get("depth", W)

# ---------------------------------------------------------------- studio
bd = SPEC.get("backdrop", [0.90, 0.90, 0.90])
bmat = bpy.data.materials.new("backdrop")
nt = _nodes(bmat)
bp, _ = _principled(nt)
bp.inputs["Base Color"].default_value = (*bd, 1)
bp.inputs["Roughness"].default_value = 0.9
bp.inputs["Specular IOR Level"].default_value = 0.2

S = max(H, W)
# seamless sweep: floor -> curve -> wall
prof = []
R = 1.8 * S
yw = 2.2 * S
for k in range(0, 40):
    prof.append((-12 * S + k * (yw - R + 12 * S) / 40, 0.0))
for k in range(0, 33):
    a = -math.pi / 2 + (math.pi / 2) * k / 32
    prof.append((yw - R + R * math.cos(a), R + R * math.sin(a)))
for k in range(1, 20):
    prof.append((yw, R + k * 12 * S / 20))
verts, faces = [], []
xs = [-14 * S + i * 28 * S / 30 for i in range(31)]
for (y, z) in prof:
    for x in xs:
        verts.append((x, y, z))
nx = len(xs)
for i in range(len(prof) - 1):
    for j in range(nx - 1):
        a = i * nx + j
        faces.append((a, a + 1, a + nx + 1, a + nx))
bdo = _finish("backdrop", verts, faces, bmat)

# camera
elev = math.radians(SPEC.get("elev", 9.0))
focal = SPEC.get("focal", 90.0)
frac = SPEC.get("frame", 0.74)
fov = 2 * math.atan(18.0 / focal)
target = Vector((0, 0, H * 0.5))
extent = max(H, W * 1.02)
dist = (extent / frac) / (2 * math.tan(fov / 2)) + Dp * 0.5
cam_data = bpy.data.cameras.new("cam")
cam_data.lens = focal
cam_data.sensor_width = 36
cam_data.dof.use_dof = True
cam_data.dof.aperture_fstop = 5.6
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
cam.location = target + Vector((0, -dist * math.cos(elev), dist * math.sin(elev)))
cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
cam_data.dof.focus_distance = (target - cam.location).length - Dp * 0.25
scene.camera = cam


def area(name, loc, size, energy, color=(1, 1, 1), size_y=None):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy
    ld.color = color
    if size_y:
        ld.shape = "RECTANGLE"
        ld.size = size
        ld.size_y = size_y
    else:
        ld.shape = "DISK"
        ld.size = size
    lo = bpy.data.objects.new(name, ld)
    scene.collection.objects.link(lo)
    lo.location = Vector(loc)
    lo.rotation_euler = (target - lo.location).to_track_quat("-Z", "Y").to_euler()
    return lo


L = SPEC.get("light", {})
ks = (S / 0.35) ** 2 * L.get("gain", 1.0) * SPEC.get("energy", 0.04)
side = L.get("side", -1)  # key from left by default
area("key", (side * 2.0 * S, -1.9 * S, 2.1 * S), 1.6 * S, 200 * ks, (1.0, 0.985, 0.96), 2.2 * S)
area("fill", (-side * 2.6 * S, -1.6 * S, 0.9 * S), 2.6 * S, 70 * ks, (0.96, 0.98, 1.0))
area("top", (0.3 * S, 0.4 * S, 3.2 * S), 2.0 * S, 120 * ks)
area("rim", (-side * 1.4 * S, 2.0 * S, 1.6 * S), 1.0 * S, 110 * ks)
# light that only washes the backdrop (sits above frame, aims at the sweep)
wl = area("wall", (0, 0.9 * S, 2.9 * S), 3.0 * S, L.get("back", 420) * ks, size_y=1.6 * S)
wl.rotation_euler = (Vector((0, 2.3 * S, 0.7 * S)) - wl.location).to_track_quat("-Z", "Y").to_euler()
for lo in [o for o in scene.objects if o.type == "LIGHT"]:
    lo.visible_camera = False

if SPEC.get("glass"):
    # black flags either side: give clear glass the dark edge reflections of a real glass shoot
    fm = bpy.data.materials.new("flag")
    fnt = _nodes(fm)
    fp, _ = _principled(fnt)
    fp.inputs["Base Color"].default_value = (0.01, 0.01, 0.01, 1)
    fp.inputs["Roughness"].default_value = 0.8
    fx = SPEC.get("flag_x", 0.95)
    for sx in (-1, 1):
        for (xa, y0f, y1f) in ((1.5, -0.6, 0.6), (fx, 0.25, 2.0)):
            fv = [(sx * xa * S, y0f * S, -0.1 * S), (sx * xa * S, y1f * S, -0.1 * S),
                  (sx * xa * S, y1f * S, 3 * S), (sx * xa * S, y0f * S, 3 * S)]
            fo = _finish(f"flag{sx}{xa}", fv, [(0, 1, 2, 3)], fm, smooth=False)
            fo.visible_camera = False
            fo.visible_shadow = False

world = bpy.data.worlds.new("w")
scene.world = world
wn = _nodes(world)
env = wn.nodes.new("ShaderNodeTexEnvironment")
env.image = bpy.data.images.load(os.path.join(bpy.utils.resource_path("LOCAL"), "datafiles", "studiolights", "world", SPEC.get("hdri", "studio.exr")))
bgn = wn.nodes.new("ShaderNodeBackground")
bgn.inputs["Strength"].default_value = L.get("env", 0.35)
wo = wn.nodes.new("ShaderNodeOutputWorld")
wn.links.new(env.outputs["Color"], bgn.inputs["Color"])
wn.links.new(bgn.outputs["Background"], wo.inputs["Surface"])

# ---------------------------------------------------------------- render photo
r = scene.render
r.resolution_x = RES
r.resolution_y = RES
r.resolution_percentage = 100
r.image_settings.file_format = "PNG"
r.image_settings.color_mode = "RGB"
r.image_settings.color_depth = "16"
r.engine = "CYCLES"
c = scene.cycles
c.device = "CPU"
c.samples = SAMPLES
c.use_adaptive_sampling = True
c.adaptive_threshold = 0.02
c.adaptive_min_samples = 16
scene.view_layers[0].cycles.denoising_store_passes = False
c.use_denoising = True
c.denoiser = "OPENIMAGEDENOISE"
c.max_bounces = 12
c.diffuse_bounces = 2
c.glossy_bounces = 4
c.transmission_bounces = 16
c.transparent_max_bounces = 8
c.caustics_reflective = False
c.caustics_refractive = False
c.blur_glossy = 0.6
c.sample_clamp_indirect = 8.0
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = SPEC.get("look", "AgX - Punchy")
scene.view_settings.exposure = L.get("exposure", 0.8)
r.filepath = os.path.join(OUT, "photo.png")
if not SPEC.get("skip_photo"):
    bpy.ops.render.render(write_still=True)

# ---------------------------------------------------------------- label pass
r.engine = "BLENDER_WORKBENCH"
sh = scene.display.shading
sh.light = "FLAT"
sh.color_type = "OBJECT"
sh.show_object_outline = False
sh.show_cavity = False
sh.show_shadows = False
sh.show_specular_highlight = False
sh.show_xray = False
scene.display.render_aa = "OFF"
r.film_transparent = True
r.dither_intensity = 0.0
r.image_settings.color_mode = "RGBA"
r.image_settings.color_depth = "8"
scene.view_settings.view_transform = "Raw"
scene.view_settings.look = "None"
scene.view_settings.exposure = 0.0
bdo.hide_render = True
bdo.hide_viewport = True
for fo in [o for o in scene.objects if o.name.startswith("flag")]:
    fo.hide_render = True
    fo.hide_viewport = True
cam_data.dof.use_dof = False
for ob in parts:
    lab = int(ob["label"])
    ob.color = (lab * 8 / 255.0, 0.5, 0.25, 1.0)
if os.environ.get("LABEL_ENGINE", "CYCLES") == "CYCLES":
    # headless-safe label pass (no OpenGL needed): flat emission colours, 1 sample, no pixel filter
    for ob in parts:
        lab = int(ob["label"])
        em = bpy.data.materials.new(f"lab{lab}_{ob.name}")
        ent = _nodes(em)
        eo = ent.nodes.new("ShaderNodeOutputMaterial")
        ee = ent.nodes.new("ShaderNodeEmission")
        ee.inputs["Color"].default_value = (lab * 8 / 255.0, 0.5, 0.25, 1.0)
        ee.inputs["Strength"].default_value = 1.0
        ent.links.new(ee.outputs[0], eo.inputs["Surface"])
        ob.data.materials.clear()
        ob.data.materials.append(em)
        ob.material_slots[0].link = "OBJECT"
        ob.material_slots[0].material = em
    for lo in [o for o in scene.objects if o.type == "LIGHT"]:
        lo.hide_render = True
    scene.world = None
    r.engine = "CYCLES"
    c.samples = 1
    c.use_adaptive_sampling = False
    c.use_denoising = False
    c.max_bounces = 0
    c.filter_width = 0.01
    c.seed = 0
r.filepath = os.path.join(OUT, "labels.png")
bpy.ops.render.render(write_still=True)

# ---------------------------------------------------------------- geometry json
def proj(p):
    v = world_to_camera_view(scene, cam, Vector(p))
    return [v.x * RES, (1 - v.y) * RES, v.z]


dg = bpy.context.evaluated_depsgraph_get()
camloc = cam.matrix_world.translation
tol = 0.006 * S


def visible(p):
    p = Vector(p)
    d = p - camloc
    L = d.length
    hit, loc, *_ = scene.ray_cast(dg, camloc, d.normalized(), distance=L + tol)
    if not hit:
        return 1
    return 1 if (loc - camloc).length >= L - tol else 0


rings_out = []
for rg in SPEC.get("rings", []):
    rr, zz = rg["r"], rg["z"]
    ox, oy = rg.get("offset", (0, 0))
    pts = []
    for k in range(181):
        a = 2 * math.pi * k / 180
        x, y = ox + rr * math.cos(a), oy + rr * math.sin(a)
        q = proj((x, y, zz))
        pts.append([q[0], q[1], 1 if y < oy else 0, visible((x, y, zz))])  # front = toward camera (-y)
    rings_out.append({"kind": rg.get("kind", "guide"), "pts": pts})

axis = [proj((0, 0, -0.02 * H)), proj((0, 0, H * 1.04))]
geo = {"res": RES, "rings": rings_out, "axis": axis,
       "base_center": proj((0, 0, 0)), "top_center": proj((0, 0, H))}
json.dump(geo, open(os.path.join(OUT, "geo.json"), "w"))
print("DONE", OUT)
