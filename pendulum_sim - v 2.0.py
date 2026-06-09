"""
N-Pendulum Simulator
Single to 7 pendulums — length, grid, color, continuous motion trails.
Features: configurable initial angle (-180°~180°), dynamic auto-zoom view.
Author: [CesiumAuride526]
License: MIT (see LICENSE file in repository)
Copyright (c) 2026 [CesiumAuride526]
Depends on: numpy, matplotlib
"""



import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons, TextBox
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle
import json
import os
from collections import deque
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

# ─── Physics ──────────────────────────────────────────

def build_mass_matrix(theta, lengths, n):
    M = np.zeros((n, n))
    for p in range(n):
        for q in range(n):
            k = max(p, q)
            M[p, q] = lengths[p] * lengths[q] * np.cos(theta[p] - theta[q]) * (n - k)
    return M

def build_force(theta, omega, lengths, g, n):
    f = np.zeros(n)
    for p in range(n):
        coupling = 0.0
        for q in range(n):
            if q == p: continue
            k = max(p, q)
            coupling += lengths[q] * np.sin(theta[p] - theta[q]) * omega[q]**2 * (n - k)
        f[p] = -lengths[p] * (coupling + g * np.sin(theta[p]) * (n - p))
    return f

def n_pendulum_derivatives(t, state, lengths, g):
    n = len(lengths)
    theta, omega = state[:n], state[n:]
    M = build_mass_matrix(theta, lengths, n)
    f = build_force(theta, omega, lengths, g, n)
    alpha = np.linalg.solve(M, f)
    dst = np.zeros(2 * n)
    dst[:n], dst[n:] = omega, alpha
    return dst

def rk4_step(f, state, t, dt, lengths, g):
    k1 = f(t, state, lengths, g)
    k2 = f(t + dt/2, state + dt/2*k1, lengths, g)
    k3 = f(t + dt/2, state + dt/2*k2, lengths, g)
    k4 = f(t + dt, state + dt*k3, lengths, g)
    return state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def get_bob_positions(theta, lengths):
    n = len(theta)
    xs, ys = np.zeros(n), np.zeros(n)
    x = y = 0.0
    for i in range(n):
        x += lengths[i] * np.sin(theta[i])
        y -= lengths[i] * np.cos(theta[i])
        xs[i], ys[i] = x, y
    return xs, ys

# ─── Defaults ─────────────────────────────────────────

COLORS = ['#66ccff','#EE0000','#2ecc71','#f39c12',
          '#9b59b6','#1abc9c','#e67e22']

# ─── Simulator ────────────────────────────────────────

class PendulumSim:
    def __init__(self):
        self.n = 1
        self.lengths = [5.0]
        self.g, self.dt = 9.81, 0.02
        self.speed_factor = 1.0
        self.state = np.zeros(2); self.state[0] = np.pi * 0.45
        self.t = 0.0

        self.grid_step = 1
        self.show_grid = True
        self.bg_color = '#2b2b2b'
        self.grid_color = '#55aa66'
        self.trail_length = 120
        self.initial_angle_deg = 81.0   # 默认角度 ≈ 0.45π
        self.auto_zoom = True           # 动态缩放开关
        # trail 在 _update 中每帧重建

        self.pendulum_colors = list(COLORS)
        self.trail_colors = list(COLORS)

        self.trail_hist = deque(maxlen=self.trail_length)
        self.paused = False
        self.anim = None
        self._build_ui()

    # ─── Build UI ─────────────────────────────────────

    def _build_ui(self):
        self.fig = plt.figure(figsize=(16, 10), facecolor=self.bg_color)
        self.fig.canvas.manager.set_window_title('N-Pendulum Simulator')

        # Plot area: left 52%
        self.ax = plt.axes([0.02, 0.04, 0.46, 0.93])
        self.ax.set_aspect('equal')
        self.ax.set_facecolor(self.bg_color)
        self._rescale()

        # ── Right panel ──
        px  = 0.55           # panel left
        pw  = 0.42           # panel width
        y0  = 0.95
        rh  = 0.026          # row height
        gap = 0.006          # spacing between rows

        def row(i):
            return y0 - (i + 1) * (rh + gap)

        def wdg(x, y, w, h=rh):
            return plt.axes([x, y, w, h])

        def sec_label(text, y_pos, x_pos=None):
            if x_pos is None:
                x_pos = sx
            self.fig.text(x_pos, y_pos + rh + 0.012, text,
                          fontsize=8, fontweight='bold', color='#7f8c8d',
                          ha='left', va='bottom')

        sw_full = pw * 0.65      # standalone slider width
        sx = px + (pw - sw_full) / 2  # centered x

        idx = 0

        # ═══════════════ PENDULUM ═══════════════
        sec_label('PENDULUM', row(idx))

        # Row 0: N slider
        a = wdg(sx, row(idx), sw_full)
        self.sl_n = Slider(a, 'Count', 1, 7, valinit=self.n, valstep=1, color='#3498db')
        self.sl_n.on_changed(self._on_n_change)
        idx += 1

        # Rows 1-7: per-pendulum (length slider + color)
        self.len_sl = []
        self.pc_boxes = []

        sw2 = sw_full * 0.75           # narrower slider to make room for color box
        tw2 = sw_full * 0.22           # color textbox width
        sep2 = 0.006

        for i in range(7):
            a = wdg(sx, row(idx), sw2)
            v = self.lengths[0] if i == 0 else 5.0
            s = Slider(a, f'L{i+1}', 1, 15, valinit=v, valstep=0.5, color=COLORS[i])
            s.on_changed(self._on_length_change)
            s.ax.set_visible(i < self.n)
            self.len_sl.append(s)

            a = wdg(sx + sw2 + sep2, row(idx), tw2, rh*0.85)
            tb = TextBox(a, f'C{i+1}', initial=COLORS[i])
            tb.ax.set_visible(i < self.n)
            tb.on_submit(self._make_color_cb(i))
            self.pc_boxes.append(tb)
            idx += 1
        idx += 1  # spacer

        # ═══════════════ PHYSICS ═══════════════
        sec_label('PHYSICS', row(idx))

        # Row: Gravity
        a = wdg(sx, row(idx), sw_full)
        self.sl_gravity = Slider(a, 'Gravity (g)', 0.5, 30.0, valinit=self.g, valstep=0.1, color='#2ecc71')
        self.sl_gravity.on_changed(self._on_gravity_change)
        idx += 1

        # Row: Init angle
        a = wdg(sx, row(idx), sw_full)
        self.sl_angle = Slider(a, 'Init angle', -180, 180, valinit=self.initial_angle_deg, valstep=1, color='#e67e22')
        self.sl_angle.on_changed(self._on_angle_change)
        idx += 1

        # spacer
        idx += 1

        # ═══════════════ DISPLAY ═══════════════
        sec_label('DISPLAY', row(idx), x_pos=px)
        hw = pw * 0.46

        # Row: Grid step + Auto zoom
        a = wdg(px, row(idx), hw)
        self.sl_step = Slider(a, 'Grid step', 1, 10, valinit=self.grid_step, valstep=1, color='#9b59b6')
        self.sl_step.on_changed(self._on_step_change)

        a = wdg(px + pw*0.50, row(idx), hw, rh*1.5)
        self.chk_zoom = CheckButtons(a, ['Auto zoom'], [self.auto_zoom])
        self.chk_zoom.on_clicked(self._on_zoom_toggle)
        idx += 1

        # Row: Show grid + Trail
        a = wdg(px, row(idx)-0.001, hw*0.6, rh*1.5)
        self.grid_btn = Button(a, '[X] Grid' if self.show_grid else '[ ] Grid')
        self.grid_btn.on_clicked(self._on_grid_toggle)
        self.grid_btn.label.set_fontsize(7)

        a = wdg(px + pw*0.50, row(idx), hw)
        self.sl_trail = Slider(a, 'Trail len', 20, 250, valinit=self.trail_length, valstep=1, color='#1abc9c')
        self.sl_trail.on_changed(self._on_trail_change)
        idx += 1

        # Row: BG + Grid color
        a = wdg(px, row(idx), hw, rh*0.85)
        self.bg_box = TextBox(a, 'BG color', initial=self.bg_color)
        self.bg_box.on_submit(self._on_bg_change)

        a = wdg(px + pw*0.50, row(idx), hw, rh*0.85)
        self.gc_box = TextBox(a, 'Grid color', initial=self.grid_color)
        self.gc_box.on_submit(self._on_grid_color_change)
        idx += 1

        # spacer
        idx += 1

        # ═══════════════ SIMULATION ═══════════════
        sec_label('SIMULATION', row(idx))

        a = wdg(sx, row(idx), sw_full)
        self.sl_speed = Slider(a, 'Speed', 0.1, 5.0, valinit=self.speed_factor, valstep=0.1, color='#e74c3c')
        self.sl_speed.on_changed(self._on_speed_change)
        idx += 1
        idx += 1  # spacer before buttons

        bw = pw * 0.30
        gap_b = pw * 0.03
        a = wdg(px, row(idx)-0.002, bw, rh*1.5)
        self.reset_btn = Button(a, 'Reset')
        self.reset_btn.on_clicked(self._on_reset)
        self.reset_btn.label.set_fontsize(7)

        a = wdg(px + bw + gap_b, row(idx)-0.002, bw, rh*1.5)
        self.save_btn = Button(a, 'Save')
        self.save_btn.on_clicked(self._on_save_config)
        self.save_btn.label.set_fontsize(7)

        a = wdg(px + (bw + gap_b) * 2, row(idx)-0.002, bw, rh*1.5)
        self.pause_btn = Button(a, 'Pause')
        self.pause_btn.on_clicked(self._on_pause)
        self.pause_btn.label.set_fontsize(7)

        self.fig.text(0.515, 0.005, 'Space=pause  R=reset',
                      fontsize=7, color='#95a5a6', style='italic')

        self.pivot = None
        self.rod_lines  = []
        self.bob_patches = []
        self.trail_artists = []

        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        self._init_physics()
        self._init_artists()          # 创建持久化 artist（不再每帧 cla 重建）
        self.anim = FuncAnimation(self.fig, self._update,
                                  interval=1000//60, blit=False,
                                  cache_frame_data=False)

        # 自动加载配置
        try:
            cfg_path = self._config_path()
            if os.path.exists(cfg_path):
                with open(cfg_path) as f:
                    cfg = json.load(f)
                self._apply_config(cfg)
        except Exception as e:
            print(f'Load config failed: {e}')

    def _init_physics(self):
        angle_rad = np.radians(self.initial_angle_deg)
        self.state = np.zeros(2 * self.n)
        for i in range(self.n):
            self.state[i] = angle_rad - i * 0.06
        self.t = 0.0
        self.trail_hist.clear()
        self._rescale()

    def _rescale(self):
        total_L = sum(self.lengths[:self.n])
        if self.auto_zoom and self.n > 0:
            # 动态缩放：根据摆锤当前位置自动调整视野
            theta = self.state[:self.n]
            xs, ys = get_bob_positions(theta, self.lengths[:self.n])
            all_x = [0.0] + list(xs)
            all_y = [0.0] + list(ys)
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)
            range_x = max(x_max - x_min, 1.0)
            range_y = max(y_max - y_min, 1.0)
            # 较大余量 + 基于摆长的绝对边距，使缩放更平滑
            margin_x = range_x * 0.5 + total_L * 0.3
            margin_y = range_y * 0.5 + total_L * 0.3
            self.ax.set_xlim(x_min - margin_x, x_max + margin_x)
            self.ax.set_ylim(y_min - margin_y, y_max + margin_y)
        else:
            total = sum(self.lengths[:self.n])
            m = total * 0.15
            self.ax.set_xlim(-total - m, total + m)
            self.ax.set_ylim(-total - m, total + m)

    # ─── Callbacks ─────────────────────────────────────

    def _on_n_change(self, val):
        self.n = int(val)
        while len(self.lengths) < self.n:
            self.lengths.append(5.0)
        self.lengths = self.lengths[:self.n]

        self.trail_hist.clear()
        self._init_physics()

        for i, sl in enumerate(self.len_sl):
            vis = i < self.n
            sl.ax.set_visible(vis)
            if vis and i < len(self.lengths):
                sl.set_val(self.lengths[i])

        for i, tb in enumerate(self.pc_boxes):
            tb.ax.set_visible(i < self.n)

        self._init_artists()
        self.fig.canvas.draw_idle()

    def _on_length_change(self, val):
        for i, sl in enumerate(self.len_sl):
            if i < self.n:
                self.lengths[i] = sl.val
        self._rescale()
        self._init_artists()
        self.fig.canvas.draw_idle()

    def _make_color_cb(self, idx):
        def cb(text):
            try:
                c = text.strip()
                mcolors.to_rgba(c)
                self.pendulum_colors[idx] = c
                self.trail_colors[idx] = c
                self.fig.canvas.draw_idle()
            except Exception:
                pass
        return cb

    def _on_step_change(self, val):
        self.grid_step = int(val)
        self._redraw()

    def _on_bg_change(self, text):
        try:
            c = text.strip(); mcolors.to_rgba(c)
            self.bg_color = c
            self.fig.set_facecolor(c); self.ax.set_facecolor(c)
            self._redraw()
        except Exception:
            pass

    def _on_grid_color_change(self, text):
        try:
            c = text.strip(); mcolors.to_rgba(c)
            self.grid_color = c
            self._redraw()
        except Exception:
            pass

    def _on_grid_toggle(self, event):
        self.show_grid = not self.show_grid
        self.grid_btn.label.set_text('[X] Grid' if self.show_grid else '[ ] Grid')
        self.ax.grid(False)
        if self.show_grid:
            self.ax.grid(True, alpha=0.35, color=self.grid_color)
        self.fig.canvas.draw()

    def _on_trail_change(self, val):
        self.trail_length = int(val)
        items = list(self.trail_hist)
        self.trail_hist = deque(items, maxlen=self.trail_length)
        self.fig.canvas.draw_idle()

    def _on_angle_change(self, val):
        self.initial_angle_deg = val
        self._init_physics()
        self.trail_hist.clear()
        self.paused = False
        self.pause_btn.label.set_text('Pause')
        self._init_artists()
        self.fig.canvas.draw_idle()

    def _on_zoom_toggle(self, label):
        self.auto_zoom = not self.auto_zoom
        self.fig.canvas.draw_idle()

    def _on_gravity_change(self, val):
        self.g = val
        self.fig.canvas.draw_idle()

    def _on_speed_change(self, val):
        self.speed_factor = val
        self.fig.canvas.draw_idle()

    def _config_path(self):
        return os.path.join(os.path.dirname(__file__), 'pendulum_config.json')

    def _on_save_config(self, event):
        cfg = {
            'n': self.n,
            'lengths': self.lengths[:self.n],
            'g': self.g,
            'speed_factor': self.speed_factor,
            'grid_step': self.grid_step,
            'show_grid': self.show_grid,
            'bg_color': self.bg_color,
            'grid_color': self.grid_color,
            'trail_length': self.trail_length,
            'initial_angle_deg': self.initial_angle_deg,
            'auto_zoom': self.auto_zoom,
            'pendulum_colors': self.pendulum_colors[:self.n],
            'trail_colors': self.trail_colors[:self.n],
        }
        try:
            with open(self._config_path(), 'w') as f:
                json.dump(cfg, f, indent=2)
            self.save_btn.label.set_text('Saved!')
            self.fig.canvas.draw_idle()
        except Exception as e:
            print(f'Save config failed: {e}')

    def _apply_config(self, cfg):
        """从 dict 加载配置并同步所有 UI 控件."""
        self.n = cfg.get('n', 1)
        self.lengths = list(cfg.get('lengths', [5.0]))
        self.g = cfg.get('g', 9.81)
        self.speed_factor = cfg.get('speed_factor', 1.0)
        self.grid_step = cfg.get('grid_step', 1)
        self.show_grid = cfg.get('show_grid', True)
        self.bg_color = cfg.get('bg_color', '#2b2b2b')
        self.grid_color = cfg.get('grid_color', '#55aa66')
        self.trail_length = cfg.get('trail_length', 120)
        self.initial_angle_deg = cfg.get('initial_angle_deg', 81.0)
        self.auto_zoom = cfg.get('auto_zoom', True)
        cols = cfg.get('pendulum_colors', COLORS)
        self.pendulum_colors = list(cols) + COLORS[len(cols):]
        self.trail_colors = list(cfg.get('trail_colors', cols)) + COLORS[len(cols):]

        # 更新 trail_hist maxlen
        self.trail_hist = deque(self.trail_hist, maxlen=self.trail_length)

        # 同步 UI 控件值（不触发额外 init_physics/init_artists）
        self.sl_n.set_val(self.n)
        for i, sl in enumerate(self.len_sl):
            if i < self.n:
                sl.set_val(self.lengths[i])
        self.sl_gravity.set_val(self.g)
        self.sl_angle.set_val(self.initial_angle_deg)
        self.sl_step.set_val(self.grid_step)
        self.sl_trail.set_val(self.trail_length)
        self.sl_speed.set_val(self.speed_factor)

        # 颜色文本框
        for i, tb in enumerate(self.pc_boxes):
            if i < self.n:
                tb.set_val(self.pendulum_colors[i])

        # 复选框状态
        active = self.chk_zoom.get_status()
        if active[0] != self.auto_zoom:
            self.chk_zoom.set_active(0)
        self.grid_btn.label.set_text('[X] Grid' if self.show_grid else '[ ] Grid')

        # 背景/网格色
        self.fig.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)

        # 最终重建
        self.trail_hist.clear()
        self._init_physics()
        self._init_artists()
        self.fig.canvas.draw_idle()

    def _on_reset(self, event):
        self._init_physics(); self.trail_hist.clear()
        self.paused = False
        self.pause_btn.label.set_text('Pause')
        self._init_artists()
        self.fig.canvas.draw_idle()

    def _on_pause(self, event):
        self.paused = not self.paused
        self.pause_btn.label.set_text('Run' if self.paused else 'Pause')

    def _on_key(self, event):
        if event.key == ' ': self._on_pause(event)
        elif event.key == 'r': self._on_reset(event)

    # ─── Drawing ───────────────────────────────────────

    def _clear_all_artists(self):
        """移除所有动态 artist（轨迹 + 摆锤），保留网格."""
        for art in list(self.trail_artists):
            try:
                art.remove()
            except Exception:
                pass
        self.trail_artists.clear()
        for l in list(self.rod_lines):
            try:
                l.remove()
            except Exception:
                pass
        self.rod_lines.clear()
        for p in list(self.bob_patches):
            try:
                p.remove()
            except Exception:
                pass
        self.bob_patches.clear()
        if self.pivot:
            try:
                self.pivot.remove()
            except Exception:
                pass
        self.pivot = None

    def _init_artists(self):
        """创建持久化 artist：网格 + 轨迹 + 摆锤。首次启动和重置时调用."""
        self._clear_all_artists()
        self.ax.set_aspect('equal')
        for spine in self.ax.spines.values():
            spine.set_color('#888888')
        self.ax.tick_params(colors='#cccccc')
        self.ax.xaxis.set_major_locator(ticker.MultipleLocator(self.grid_step))
        self.ax.yaxis.set_major_locator(ticker.MultipleLocator(self.grid_step))
        # Always turn grid off first, then only on with styling
        self.ax.grid(False)
        if self.show_grid:
            self.ax.grid(True, alpha=0.35, color=self.grid_color)
        self._rebuild_trails()
        self._rebuild_pendulum_artists()

    def _rebuild_trails(self):
        """重建轨迹 PolyCollection + LineCollection.
        Draw continuous swept surface: filled quadrilaterals between
        consecutive pendulum frames forming a gradient swept-area effect.
        Each rod sweeps a quadrilateral between frame k and k+1, creating
        a continuous fan/surface instead of discrete lines.
        """
        # Remove old trail artists
        for art in list(self.trail_artists):
            try:
                art.remove()
            except Exception:
                pass
        self.trail_artists.clear()

        hist = list(self.trail_hist)
        if len(hist) < 4:
            return

        nf = len(hist)
        n = self.n

        # ─── Swept surface: filled quadrilaterals per rod ───
        for j in range(n):
            quads = []
            for k in range(nf - 1):
                xs_k, ys_k = hist[k]
                xs_k1, ys_k1 = hist[k+1]

                # Pivot point (where rod attaches)
                if j == 0:
                    px_k, py_k   = 0.0, 0.0
                    px_k1, py_k1 = 0.0, 0.0
                else:
                    px_k, py_k   = xs_k[j-1], ys_k[j-1]
                    px_k1, py_k1 = xs_k1[j-1], ys_k1[j-1]

                bx_k, by_k   = xs_k[j], ys_k[j]
                bx_k1, by_k1 = xs_k1[j], ys_k1[j]

                # Quadrilateral: pivot_k → bob_k → bob_k+1 → pivot_k+1
                quads.append(np.array([
                    [px_k, py_k],
                    [bx_k, by_k],
                    [bx_k1, by_k1],
                    [px_k1, py_k1],
                ]))

            if not quads:
                continue

            tc = mcolors.to_rgba(self.trail_colors[j])
            alphas = np.linspace(0.01, 0.25, nf - 1)
            colors = np.array([[*tc[:3], a] for a in alphas])

            pc = PolyCollection(quads, facecolors=colors,
                               edgecolors='none', zorder=0)
            self.ax.add_collection(pc)
            self.trail_artists.append(pc)

        # ─── Bob trajectory line (edge emphasis) ───
        for j in range(n):
            pts = np.array([(hist[k][0][j], hist[k][1][j]) for k in range(nf)])
            segs = np.array([pts[k:k+2] for k in range(nf - 1)])
            tc = mcolors.to_rgba(self.trail_colors[j])
            alphas = np.linspace(0.03, 0.55, nf - 1)
            colors = np.array([[*tc[:3], a] for a in alphas])
            lc = LineCollection(segs, colors=colors, linewidth=1.5,
                               capstyle='round', zorder=1)
            self.ax.add_collection(lc)
            self.trail_artists.append(lc)

    def _update_trails(self):
        """原地更新轨迹顶点/颜色，不创建新 collection 对象."""
        hist = list(self.trail_hist)
        if len(hist) < 4:
            return

        nf = len(hist)
        n_poly = n = min(self.n, len(hist[0][0]))
        n_line = n
        expected = n_poly + n_line

        # 如果 collection 数量不匹配，回退到完整重建
        if len(self.trail_artists) != expected:
            self._rebuild_trails()
            return

        # ─── PolyCollection: swept surface ───
        for j in range(n_poly):
            quads = []
            for k in range(nf - 1):
                xs_k, ys_k = hist[k]
                xs_k1, ys_k1 = hist[k+1]
                if j == 0:
                    px_k, py_k = 0.0, 0.0
                    px_k1, py_k1 = 0.0, 0.0
                else:
                    px_k, py_k = xs_k[j-1], ys_k[j-1]
                    px_k1, py_k1 = xs_k1[j-1], ys_k1[j-1]
                bx_k, by_k = xs_k[j], ys_k[j]
                bx_k1, by_k1 = xs_k1[j], ys_k1[j]
                quads.append(np.array([
                    [px_k, py_k], [bx_k, by_k],
                    [bx_k1, by_k1], [px_k1, py_k1],
                ]))

            pc = self.trail_artists[j]
            pc.set_verts(quads)
            tc = mcolors.to_rgba(self.trail_colors[j])
            alphas = np.linspace(0.01, 0.25, nf - 1)
            pc.set_facecolors(np.array([[*tc[:3], a] for a in alphas]))
            pc.set_edgecolors('none')

        # ─── LineCollection: edge emphasis ───
        for j in range(n_line):
            pts = np.array([(hist[k][0][j], hist[k][1][j]) for k in range(nf)])
            segs = np.array([pts[k:k+2] for k in range(nf - 1)])
            lc = self.trail_artists[n_poly + j]
            lc.set_segments(segs)
            tc = mcolors.to_rgba(self.trail_colors[j])
            alphas = np.linspace(0.03, 0.55, nf - 1)
            lc.set_colors(np.array([[*tc[:3], a] for a in alphas]))
            lc.set_linewidth(1.5)

    def _rebuild_pendulum_artists(self):
        """重建持久化摆锤 artist（杆线 + 圆盘 + 悬点）."""
        for l in list(self.rod_lines):
            try:
                if l in self.ax.lines: l.remove()
            except Exception:
                pass
        self.rod_lines.clear()
        for p in list(self.bob_patches):
            try:
                if p in self.ax.patches: p.remove()
            except Exception:
                pass
        self.bob_patches.clear()
        if self.pivot and self.pivot in self.ax.lines:
            try:
                self.pivot.remove()
            except Exception:
                pass
        self.pivot = None

        if self.n == 0:
            return

        theta = self.state[:self.n]
        xs, ys = get_bob_positions(theta, self.lengths[:self.n])

        self.pivot = self.ax.plot(0, 0, 'o', color='#333', ms=7, zorder=6)[0]
        px = py = 0.0
        for i in range(self.n):
            c = self.pendulum_colors[i]
            l, = self.ax.plot([px, xs[i]], [py, ys[i]], color=c, lw=2.5, zorder=3)
            self.rod_lines.append(l)
            circ = Circle((xs[i], ys[i]), 0.4, facecolor=c, edgecolor='#333',
                         lw=1.5, zorder=4)
            self.ax.add_patch(circ)
            self.bob_patches.append(circ)
            px, py = xs[i], ys[i]

    def _update_pendulum_inplace(self):
        """原地更新摆锤 artist 坐标（不创建新对象）."""
        if self.n == 0:
            return
        theta = self.state[:self.n]
        xs, ys = get_bob_positions(theta, self.lengths[:self.n])
        px, py = 0.0, 0.0
        for i in range(self.n):
            if i < len(self.rod_lines):
                self.rod_lines[i].set_data([px, xs[i]], [py, ys[i]])
            if i < len(self.bob_patches):
                self.bob_patches[i].center = (xs[i], ys[i])
            px, py = xs[i], ys[i]

    def _redraw(self):
        """全量重建：网格 + 轨迹 + 摆锤（重置/N变化时调用）."""
        self.ax.cla()
        self._rescale()
        self._init_artists()
        self.fig.canvas.draw_idle()

    def _update(self, frame):
        try:
            if not self.paused:
                n_steps = max(1, int(2 * self.speed_factor))
                for _ in range(n_steps):
                    self.state = rk4_step(n_pendulum_derivatives,
                                          self.state, self.t, self.dt,
                                          self.lengths[:self.n], self.g)
                    self.t += self.dt
                    theta = self.state[:self.n]
                    if self.n > 0:
                        xs, ys = get_bob_positions(theta, self.lengths[:self.n])
                        self.trail_hist.append((xs.copy(), ys.copy()))

            # 原地更新摆锤（无对象创建，极快）
            self._update_pendulum_inplace()
            # 更新坐标范围
            self._rescale()
            self._update_trails()

            self.fig.canvas.draw_idle()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.paused = True
        return []

    def show(self):
        plt.show()


if __name__ == '__main__':
    PendulumSim().show()