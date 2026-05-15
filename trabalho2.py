import os
import cv2
import cv2.aruco as aruco
import numpy as np
import sys
import time
import math
import mediapipe as mp
import pygame
import urllib.request
from mediapipe.tasks import python as _mpt
from mediapipe.tasks.python import vision as _mpv
import pyrender
import trimesh

try:
    import trimesh
    import pyrender
    PYRENDER_OK = True
except Exception:
    PYRENDER_OK = False

_skull_scene = None
_skull_renderer = None
_skull_mesh_node = None


# ── Dependências do pygame e da api 3d ────────────────────────
try:
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False
    print("[AVISO] pygame não encontrado. Sons desativados.")

try:
    _MP_API = "legacy" if hasattr(mp, "solutions") else "new"
    MP_OK = True
except Exception:
    MP_OK = False
    _MP_API = None
    print("[AVISO] mediapipe não encontrado. Modo (3c) desativado.")

# ── Geração de sons ───────────────────────────────
NOTE_FREQS = {"DO": 261.63, "RE": 293.66, "MI": 329.63, "FA": 349.23, "SOL": 392.00, "LA": 440.00, "SI": 493.88}
HOLE_NOTES = ["DO", "RE", "MI", "FA", "SOL", "LA", "SI"]

def _make_tone(freq):
    sr, dur = 44100, 0.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    w = (np.sin(2 * np.pi * freq * t) * 0.6 * 32767).astype(np.int16)
    a, r = int(0.01 * sr), int(0.1 * sr)
    w[:a]  = (w[:a]  * np.linspace(0, 1, a)).astype(np.int16)
    w[-r:] = (w[-r:] * np.linspace(1, 0, r)).astype(np.int16)
    return pygame.sndarray.make_sound(w)

SOUNDS = {n: _make_tone(f) for n, f in NOTE_FREQS.items()} if PYGAME_OK else {}

# ── Configuração ArUco ────────────────────────────
ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
_params = aruco.DetectorParameters()
_params.adaptiveThreshWinSizeMin    = 3
_params.adaptiveThreshWinSizeMax    = 23
_params.adaptiveThreshWinSizeStep   = 4
_params.minMarkerPerimeterRate      = 0.02
_params.maxMarkerPerimeterRate      = 4.0
_params.polygonalApproxAccuracyRate = 0.04
_params.cornerRefinementMethod      = aruco.CORNER_REFINE_SUBPIX
ARUCO_DETECTOR = aruco.ArucoDetector(ARUCO_DICT, _params)

ID_MET_A      = 0   # metrologia marcador A
ID_MET_B      = 1   # metrologia marcador B
ID_OCR_BASE   = 7   # ocarina base
ID_HOLE_START = 0   # furos IDs 3-8

MARKER_CM = 4.1     # tamanho real do marcador impresso em cm

# ── Utilitários ───────────────────────────────────
def marker_center(corners):
    c = corners[0][0]  # shape (4, 2)
    return int(c[:, 0].mean()), int(c[:, 1].mean())

def txt(img, text, pos, scale=0.7, color=(0,255,0), thickness=2):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = ARUCO_DETECTOR.detectMarkers(gray)
    result = {}
    if ids is not None:
        for i, mid in enumerate(ids.flatten()):
            result[int(mid)] = corners[i]
    return result

# ── Modo 1: Metrologia ────────────────────────────
def mode_metrologia(cap):
    print("\n[METROLOGIA] Marcadores ID 0 e ID 1 | Q=sair\n")
    while True:
        ret, frame = cap.read()
        if not ret: break
        found = detect(frame)

        for mid in [ID_MET_A, ID_MET_B]:
            if mid in found:
                aruco.drawDetectedMarkers(frame, [found[mid]], np.array([[mid]]))

        if ID_MET_A in found and ID_MET_B in found:
            def left_edge(corners):
                c = corners[0]
                pt = ((c[0] + c[3]) / 2).astype(int)
                return (pt[0], pt[1])

            def marker_size_px(corners):
                c = corners[0]
                sides = [np.linalg.norm(c[i] - c[(i+1) % 4]) for i in range(4)]
                return np.mean(sides)

            ea = left_edge(found[ID_MET_A])
            eb = left_edge(found[ID_MET_B])

            size_a = marker_size_px(found[ID_MET_A])
            size_b = marker_size_px(found[ID_MET_B])
            avg_size = (size_a + size_b) / 2

            dist_px = math.dist(ea, eb)
            dist = dist_px * MARKER_CM / avg_size if avg_size else 0.0

            cv2.circle(frame, ea, 6, (0, 200, 255), -1)
            cv2.circle(frame, eb, 6, (0, 200, 255), -1)
            mp2 = ((ea[0]+eb[0])//2, (ea[1]+eb[1])//2)
            label = f"{dist:.1f} cm"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (mp2[0]-tw//2-4, mp2[1]-th-6), (mp2[0]+tw//2+4, mp2[1]+2), (0,0,0), -1)
            txt(frame, label, (mp2[0]-tw//2, mp2[1]-2))
        else:
            msg = "Aponte tambem o segundo marcador" if (ID_MET_A in found or ID_MET_B in found) \
                else "Aponte os marcadores ID 0 e ID 1"
            txt(frame, msg, (10, 30), color=(0, 100, 255))

        txt(frame, "METROLOGIA | Q=sair", (10, frame.shape[0]-10), scale=0.5, color=(180,180,180))
        cv2.imshow("Metrologia", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()

# ── Modo 2: Ocarina ───────────────────────────────

def draw_flute(frame, found):

    # Apenas IDs dos furos
    hole_ids = sorted([
        mid for mid in found.keys()
        if mid != ID_OCR_BASE
    ])

    if len(hole_ids) < 2:
        return

    first_id = hole_ids[0]
    last_id  = hole_ids[-1]

    p1 = marker_center([found[first_id]])
    p2 = marker_center([found[last_id]])

    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    comprimento = math.hypot(dx, dy)

    if comprimento < 10:
        return

    ux = dx / comprimento
    uy = dy / comprimento

    px = -uy
    py = ux

    largura = 30

    a = (int(p1[0] + px*largura), int(p1[1] + py*largura))
    b = (int(p1[0] - px*largura), int(p1[1] - py*largura))
    c = (int(p2[0] - px*largura), int(p2[1] - py*largura))
    d = (int(p2[0] + px*largura), int(p2[1] + py*largura))

    corpo = np.array([a, b, c, d])

    overlay = frame.copy()

    # Corpo da flauta
    cv2.fillConvexPoly(
        overlay,
        corpo,
        (70, 160, 210)
    )

    cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0,
        frame
    )

    cv2.polylines(
        frame,
        [corpo],
        True,
        (20,20,20),
        2
    )

    # Furos
    for mid in hole_ids:

        cx, cy = marker_center([found[mid]])

        cv2.circle(frame, (cx, cy), 12, (20,20,20), -1)
        cv2.circle(frame, (cx, cy), 8, (0,0,0), -1)

    # Extremidades
    cv2.circle(frame, p1, 16, (220,220,220), 2)
    cv2.circle(frame, p2, 16, (220,220,220), 2)


def mode_ocarina(cap):
    print("\n[OCARINA] Base=ID7 | Furos=ID0-6 | Cubra para tocar | Q=sair\n")

    cooldown = {}
    COOL = 0.4

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        found = detect(frame)

        # =========================
        # DESENHA A FLAUTA
        # =========================
        draw_flute(frame, found)

        # =========================
        # MARCADOR BASE
        # =========================
        if ID_OCR_BASE in found:
            aruco.drawDetectedMarkers(
                frame,
                [found[ID_OCR_BASE]],
                np.array([[ID_OCR_BASE]])
            )

            bx, by = marker_center([found[ID_OCR_BASE]])

            txt(
                frame,
                "OCARINA",
                (bx - 35, by - 20),
                scale=0.6,
                color=(255, 220, 50)
            )

        nota_ativa = None

        # =========================
        # FUROS / NOTAS
        # =========================
        for i in range(len(HOLE_NOTES)):

            hid = ID_HOLE_START + i
            note = HOLE_NOTES[i]

            # Marcador visível
            if hid in found:

                aruco.drawDetectedMarkers(
                    frame,
                    [found[hid]],
                    np.array([[hid]])
                )

                cx, cy = marker_center([found[hid]])

                txt(
                    frame,
                    note,
                    (cx - 15, cy - 20),
                    scale=0.9,
                    color=(140, 140, 140),
                    thickness=2
                )

            else:
                # Marcador tampado
                if ID_OCR_BASE in found:

                    now = time.time()

                    if PYGAME_OK and (now - cooldown.get(hid, 0)) > COOL:
                        SOUNDS[note].play()
                        cooldown[hid] = now

                    nota_ativa = note

            cor = (140, 140, 140) if hid in found else (0, 220, 100)

            sufixo = (
                " << tampado"
                if hid not in found and ID_OCR_BASE in found
                else ""
            )

            txt(
                frame,
                f"ID {hid} -> {note}{sufixo}",
                (10, 60 + i * 26),
                scale=0.48,
                color=cor,
                thickness=1
            )

        # =========================
        # STATUS
        # =========================
        status = (
            f"Tocando: {nota_ativa}"
            if nota_ativa
            else (
                "Ocarina pronta"
                if ID_OCR_BASE in found
                else f"Aponte marcador ID {ID_OCR_BASE}"
            )
        )

        txt(
            frame,
            status,
            (10, 30),
            color=(0,255,100)
            if ID_OCR_BASE in found
            else (0,100,255)
        )

        txt(
            frame,
            "OCARINA | Q=sair",
            (10, frame.shape[0]-10),
            scale=0.5,
            color=(180,180,180)
        )

        cv2.imshow("Ocarina", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# ── Modo 3: AR com MediaPipe ──────────────────────
def _cube(frame, cx, cy, size, angle):
    focal = size * 4.0
    verts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1]], dtype=np.float64)
    c, s = math.cos(angle), math.sin(angle)
    ax = 0.44; cx2, sx2 = math.cos(ax), math.sin(ax)
    verts = (np.array([[c,0,s],[0,1,0],[-s,0,c]]) @
             np.array([[1,0,0],[0,cx2,-sx2],[0,sx2,cx2]]) @ verts.T).T
    def p(v):
        z = max(v[2] + 4.0, 0.001)
        return int(cx + focal * v[0] / z), int(cy - focal * v[1] / z)
    pts = [p(v) for v in verts]
    faces = [(0,1,2,3,(100,180,255)),(4,5,6,7,(255,140,60)),
             (0,1,5,4,(80,220,120)), (2,3,7,6,(200,80,180)),
             (0,3,7,4,(220,210,60)), (1,2,6,5,(60,200,220))]
    ov = frame.copy()
    for f in sorted(faces, key=lambda f: np.mean([verts[i][2] for i in f[:4]])):
        q = np.array([pts[f[0]], pts[f[1]], pts[f[2]], pts[f[3]]])
        cv2.fillPoly(ov, [q], f[4])
        cv2.polylines(ov, [q], True, (20,20,20), 2)
    cv2.addWeighted(ov, 0.9, frame, 0.1, 0, frame)
    shadow = frame.copy()
    cv2.ellipse(shadow, (cx, cy+size+4), (max(4,size//2), max(2,size//5)), 0,0,360,(0,0,0),-1)
    cv2.addWeighted(shadow, 0.35, frame, 0.65, 0, frame)

def _draw_on_hand(frame, landmarks, w, h, angle):
    palm = [0, 5, 9, 13, 17]
    cx = int(np.mean([landmarks[i].x * w for i in palm]))
    cy = int(np.mean([landmarks[i].y * h for i in palm]))
    hand_len = math.dist((landmarks[0].x*w, landmarks[0].y*h),
                         (landmarks[9].x*w, landmarks[9].y*h))
    size = max(8, int(hand_len * 0.20))
    _cube(frame, cx, cy - int(hand_len*0.15), size, angle)
    cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

HAND_CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),
             (9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
             (13,17),(17,18),(18,19),(19,20),(0,17)]

def mode_ar_mediapipe(cap):
    if not MP_OK:
        print("[ERRO] mediapipe não instalado."); return
    print(f"\n[AR] Mostre a mão aberta | Q=sair\n")
    angle = 0.0

    if _MP_API == "legacy":
        hands = mp.solutions.hands.Hands(max_num_hands=1,
            min_detection_confidence=0.7, min_tracking_confidence=0.6)
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_hand_landmarks:
                for lms in res.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, lms, mp.solutions.hands.HAND_CONNECTIONS)
                    _draw_on_hand(frame, lms.landmark, w, h, angle)
                angle += 0.03
            else:
                txt(frame, "Mostre sua mao aberta", (10,40), color=(0,100,255))
            txt(frame, "AR | Q=sair", (10, h-10), scale=0.5, color=(180,180,180))
            cv2.imshow("Realidade Virtual", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        hands.close()
    else:
        import urllib.request
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("  Baixando modelo MediaPipe...")
            try:
                urllib.request.urlretrieve(
                    "https://storage.googleapis.com/mediapipe-models/"
                    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                    model_path)
            except Exception as e:
                print(f"  [ERRO] {e}"); return
        from mediapipe.tasks import python as _mpt
        from mediapipe.tasks.python import vision as _mpv
        det = _mpv.HandLandmarker.create_from_options(
            _mpv.HandLandmarkerOptions(
                base_options=_mpt.BaseOptions(model_asset_path=model_path),
                running_mode=_mpv.RunningMode.IMAGE, num_hands=1))
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if res.hand_landmarks:
                for lms in res.hand_landmarks:
                    pts = [(int(l.x*w), int(l.y*h)) for l in lms]
                    for a, b in HAND_CONN:
                        cv2.line(frame, pts[a], pts[b], (80,200,80), 1)
                    for pt in pts:
                        cv2.circle(frame, pt, 3, (0,255,0), -1)
                    class _L:
                        def __init__(self, l): self.x=l.x; self.y=l.y
                    _draw_on_hand(frame, [_L(l) for l in lms], w, h, angle)
                angle += 0.03
            else:
                txt(frame, "Mostre sua mao aberta", (10,40), color=(0,100,255))
            txt(frame, "AR | Q=sair", (10, h-10), scale=0.5, color=(180,180,180))
            cv2.imshow("Realidade Virtual", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        det.close()
    cv2.destroyAllWindows()

# ── Gera marcadores ───────────────────────────────
def generate_markers():
    ids = [(ID_MET_A, "marcador_met_0.png"), (ID_MET_B, "marcador_met_1.png"),
           (ID_OCR_BASE, "marcador_ocarina_base.png")]
    ids += [(ID_HOLE_START+i, f"marcador_furo_{i+1}.png") for i in range(7)]
    print("\n[MARCADORES]")
    for mid, fname in ids:
        cv2.imwrite(fname, cv2.aruco.generateImageMarker(ARUCO_DICT, mid, 200))
        print(f"  {fname}  (ID={mid})")
    print(f"\nImprima com tamanho real = {MARKER_CM} cm")
    input("ENTER para continuar...")

# ── CLI ───────────────────────────────────────────
MENU = """
╔══════════════════════════════════════════╗
║  [1] Metrologia ArUco  (IDs 0 e 1)       ║
║  [2] Ocarina Virtual   (IDs 2-8)         ║
║  [3] Realidade Virtual (MediaPipe)       ║
║  [4] Gerar marcadores ArUco (PNG)        ║
║  [0] Sair                                ║
╚══════════════════════════════════════════╝"""

def main():
    cap = None
    for idx in range(3):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            print(f"Câmera aberta (índice {idx})")
            cap = c; break
        c.release()
    if cap is None:
        try: cap = cv2.VideoCapture(int(input("Índice da câmera: ")))
        except Exception: print("Erro ao abrir câmera."); sys.exit(1)

    print(f"pygame: {'OK' if PYGAME_OK else 'desativado'} | mediapipe: {'OK' if MP_OK else 'desativado'}")
    while True:
        print(MENU)
        c = input("Opção: ").strip()
        if   c == "1": mode_metrologia(cap)
        elif c == "2": mode_ocarina(cap)
        elif c == "3": mode_ar_mediapipe(cap)
        elif c == "4": generate_markers()
        elif c == "0": print("Até logo!"); break
        else: print("Opção inválida.")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()