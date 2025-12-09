import sys
import json
import math
from functools import lru_cache

import numpy as np

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QFileDialog, QLineEdit,
    QMessageBox, QListWidget, QDoubleSpinBox
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def de_casteljau(points, t):
    P = [np.array(p, dtype=float) for p in points]
    n = len(P) - 1
    for r in range(1, n + 1):
        for i in range(n - r + 1):
            P[i] = (1 - t) * P[i] + t * P[i + 1]
    return P[0]


@lru_cache(maxsize=1024)
def cox_de_boor(i, k, t, knot_tuple):
    U = knot_tuple
    if k == 0:
        if U[i] <= t < U[i + 1] or (t == U[-1] and t == U[i + 1] and U[i] < U[i + 1]):
            return 1.0
        return 0.0
    denom1 = U[i + k] - U[i]
    denom2 = U[i + k + 1] - U[i + 1]
    term1 = 0.0
    term2 = 0.0
    if denom1 != 0:
        term1 = (t - U[i]) / denom1 * cox_de_boor(i, k - 1, t, knot_tuple)
    if denom2 != 0:
        term2 = (U[i + k + 1] - t) / denom2 * cox_de_boor(i + 1, k - 1, t, knot_tuple)
    return term1 + term2


def bspline_point(ctrl_pts, degree, knots, t):
    n = len(ctrl_pts) - 1
    required_knot_len = n + degree + 2
    if len(knots) != required_knot_len:
        raise ValueError(f"Invalid knot vector length: required {required_knot_len}, got {len(knots)}")
    knot_tuple = tuple(knots)
    point = np.zeros(2, dtype=float)
    for i in range(n + 1):
        Ni = cox_de_boor(i, degree, t, knot_tuple)
        point += Ni * np.array(ctrl_pts[i], dtype=float)
    return point


def nurbs_point(ctrl_pts, weights, degree, knots, t):
    n = len(ctrl_pts) - 1
    required_knot_len = n + degree + 2
    if len(knots) != required_knot_len:
        raise ValueError(f"Invalid knot vector length: required {required_knot_len}, got {len(knots)}")
    knot_tuple = tuple(knots)
    numerator = np.zeros(2, dtype=float)
    denom = 0.0
    for i in range(n + 1):
        Ni = cox_de_boor(i, degree, t, knot_tuple)
        wNi = weights[i] * Ni
        numerator += wNi * np.array(ctrl_pts[i], dtype=float)
        denom += wNi
    if denom == 0:
        return numerator
    return numerator / denom


def bezier_degree_elevate(ctrl_pts):
    n = len(ctrl_pts) - 1
    new = []
    new.append(np.array(ctrl_pts[0], dtype=float))
    for i in range(1, n + 1):
        alpha = i / (n + 1)
        p = alpha * np.array(ctrl_pts[i - 1], dtype=float) + (1 - alpha) * np.array(ctrl_pts[i], dtype=float)
        new.append(p)
    new.append(np.array(ctrl_pts[-1], dtype=float))
    return new


def knot_insert(ctrl_pts, degree, knots, u, weights=None):
    is_rational = weights is not None
    if is_rational:
        homo_pts = [np.array([p[0] * w, p[1] * w, w], dtype=float) for p, w in zip(ctrl_pts, weights)]
    else:
        homo_pts = [np.array([p[0], p[1], 1.0], dtype=float) for p in ctrl_pts]

    U = list(knots)
    p = degree
    n = len(ctrl_pts) - 1
    required_knot_len = n + p + 2
    if len(U) != required_knot_len:
        raise ValueError(f"Invalid knot vector length: required {required_knot_len}, got {len(U)}")

    k = 0
    for val in U:
        if abs(val - u) < 1e-9:
            k += 1
    span = -1
    for i in range(len(U) - 1):
        if U[i] <= u <= U[i + 1]:
            span = i
            break
    if span == -1:
        span = len(U) - 2
    r = 1

    Ubar = U[:span + 1] + [u] + U[span + 1:]
    Q = []
    a = span - p
    b = span
    for i in range(0, a + 1):
        Q.append(homo_pts[i])
    for i in range(a + 1, b + 1):
        alpha = (u - U[i]) / (U[i + p] - U[i]) if (U[i + p] - U[i]) != 0 else 0.0
        Pi = (1 - alpha) * homo_pts[i - 1] + alpha * homo_pts[i]
        Q.append(Pi)
    for i in range(b + 1, len(homo_pts)):
        Q.append(homo_pts[i])

    new_ctrl = []
    new_w = []
    for H in Q:
        if H[2] == 0:
            new_ctrl.append((H[0], H[1]))
            new_w.append(0.0)
        else:
            new_ctrl.append((H[0] / H[2], H[1] / H[2]))
            new_w.append(H[2])

    return new_ctrl, new_w, Ubar


class CurveObject:
    def __init__(self, kind='bezier'):
        self.kind = kind
        self.ctrl_pts = []
        self.degree = 3
        self.knots = []
        self.weights = []
        self.name = 'curve'

    def ensure_knots(self):
        if self.kind not in ('nurbs', 'bspline'):
            return
        n = len(self.ctrl_pts) - 1
        p = self.degree
        if n < 0:
            self.knots = []
            self.weights = []
            return
        required_len = n + p + 2
        if len(self.knots) != required_len:
            U = [0.0] * (p + 1)
            interior_count = required_len - 2 * (p + 1)
            if interior_count > 0:
                step = 1.0 / (interior_count + 1)
                for i in range(1, interior_count + 1):
                    U.append(i * step)
            U += [1.0] * (p + 1)
            self.knots = U
        if len(self.weights) != len(self.ctrl_pts):
            self.weights = [1.0] * len(self.ctrl_pts)

    def evaluate(self, t):
        try:
            if self.kind == 'bezier':
                return de_casteljau(self.ctrl_pts, t)
            else:
                self.ensure_knots()
                if len(self.knots) == 0 or len(self.ctrl_pts) == 0:
                    return np.zeros(2)
                if self.kind == 'bspline':
                    return bspline_point(self.ctrl_pts, self.degree, self.knots, t)
                elif self.kind == 'nurbs':
                    return nurbs_point(self.ctrl_pts, self.weights, self.degree, self.knots, t)
        except Exception as e:
            print(f"Curve evaluation error: {e}")
            return np.zeros(2)

    def sample(self, n=100):
        if len(self.ctrl_pts) == 0:
            return np.empty((0, 2))
        ts = np.linspace(0.0, 1.0, n)
        pts = []
        for t in ts:
            try:
                pt = self.evaluate(float(t))
                pts.append(pt)
            except:
                continue
        return np.array(pts)

    def to_dict(self):
        return {
            'name': self.name,
            'kind': self.kind,
            'degree': int(self.degree),
            'ctrl_pts': [[float(x), float(y)] for x, y in self.ctrl_pts],
            'knots': [float(k) for k in self.knots],
            'weights': [float(w) for w in self.weights]
        }

    @staticmethod
    def from_dict(d):
        c = CurveObject(d.get('kind', 'bezier'))
        c.name = d.get('name', 'curve')
        c.degree = int(d.get('degree', 3))
        c.ctrl_pts = [(float(x), float(y)) for x, y in d.get('ctrl_pts', [])]
        c.knots = [float(k) for k in d.get('knots', [])]
        c.weights = [float(w) for w in d.get('weights', [])]
        c.ensure_knots()
        return c


def compute_cumulative_arc_length(pts):
    if len(pts) < 2:
        return np.array([0.0])
    diffs = pts[1:] - pts[:-1]
    segment_lengths = np.sqrt(np.sum(diffs ** 2, axis=1))
    cumulative = np.cumsum(np.concatenate([[0.0], segment_lengths]))
    return cumulative


def arc_length_parameterization(pts, num_samples=100):
    cum_length = compute_cumulative_arc_length(pts)
    total_length = cum_length[-1]
    if total_length < 1e-9:
        return np.linspace(0, 1, num_samples), pts[:num_samples]

    s_target = np.linspace(0, total_length, num_samples)
    x_interp = np.interp(s_target, cum_length, pts[:, 0])
    y_interp = np.interp(s_target, cum_length, pts[:, 1])
    s_normalized = s_target / total_length
    return s_normalized, np.column_stack([x_interp, y_interp])


def fit_geometric_polynomial(pts, degree, num_samples=100):
    if len(pts) < 2 or degree < 0:
        raise ValueError("Invalid input parameters")

    s_param, pts_arc = arc_length_parameterization(pts, num_samples)
    V = np.vander(s_param, N=degree + 1, increasing=True)
    V_reg = V + 1e-8 * np.eye(V.shape[0], V.shape[1])[:V.shape[0], :V.shape[1]]

    coeffs_x, *_ = np.linalg.lstsq(V_reg, pts_arc[:, 0], rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(V_reg, pts_arc[:, 1], rcond=None)

    fit_pts = np.column_stack([
        V.dot(coeffs_x),
        V.dot(coeffs_y)
    ])

    _, pts_arc_full = arc_length_parameterization(pts, len(fit_pts))
    rmse = np.sqrt(np.mean(np.sum((pts_arc_full - fit_pts) ** 2, axis=1)))

    return s_param, fit_pts, rmse, coeffs_x, coeffs_y


def rmse_between(a, b):
    if a.shape != b.shape or a.size == 0:
        return float('nan')
    diff = a - b
    return math.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


class CurveDesigner(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Bézier & NURBS Curve Designer (Fixed Knot Vector Version)')
        self.setGeometry(100, 100, 1200, 700)

        self.curves = []
        self.current_curve = None
        self.compare_results = []
        self.ground_truth_pts = None
        self.arc_length_pts = None

        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout()
        central.setLayout(layout)

        left = QVBoxLayout()
        layout.addLayout(left, 0)

        self.curve_list = QListWidget()
        self.curve_list.currentRowChanged.connect(self.on_curve_selected)
        left.addWidget(QLabel('Curve List'))
        left.addWidget(self.curve_list)

        btn_new = QPushButton('New Curve')
        btn_new.clicked.connect(self.new_curve_dialog)
        left.addWidget(btn_new)

        btn_delete = QPushButton('Delete Curve')
        btn_delete.clicked.connect(self.delete_curve)
        left.addWidget(btn_delete)

        left.addWidget(QLabel('Curve Degree:'))
        self.spin_degree = QSpinBox()
        self.spin_degree.setRange(1, 10)
        self.spin_degree.setValue(3)
        self.spin_degree.valueChanged.connect(self.on_degree_changed)
        left.addWidget(self.spin_degree)

        left.addWidget(QLabel('Sample Density:'))
        self.spin_sample = QSpinBox()
        self.spin_sample.setRange(10, 2000)
        self.spin_sample.setValue(200)
        self.spin_sample.valueChanged.connect(self.redraw)
        left.addWidget(self.spin_sample)

        left.addWidget(QLabel('Selected Control Point Index:'))
        self.ctrl_index = QSpinBox()
        self.ctrl_index.setRange(0, 100)
        self.ctrl_index.valueChanged.connect(self.on_ctrl_index_changed)
        left.addWidget(self.ctrl_index)

        left.addWidget(QLabel('Control Point X:'))
        self.ctrl_x = QDoubleSpinBox()
        self.ctrl_x.setDecimals(4)
        self.ctrl_x.setRange(-1e6, 1e6)
        self.ctrl_x.valueChanged.connect(self.on_ctrl_coord_changed)
        left.addWidget(self.ctrl_x)

        left.addWidget(QLabel('Control Point Y:'))
        self.ctrl_y = QDoubleSpinBox()
        self.ctrl_y.setDecimals(4)
        self.ctrl_y.setRange(-1e6, 1e6)
        self.ctrl_y.valueChanged.connect(self.on_ctrl_coord_changed)
        left.addWidget(self.ctrl_y)

        left.addWidget(QLabel('Weight (NURBS only):'))
        self.ctrl_w = QDoubleSpinBox()
        self.ctrl_w.setDecimals(4)
        self.ctrl_w.setRange(0.0, 1e6)
        self.ctrl_w.setValue(1.0)
        self.ctrl_w.valueChanged.connect(self.on_weight_changed)
        left.addWidget(self.ctrl_w)

        left.addWidget(QLabel('Knot Vector (Auto-generated, Read-only):'))
        self.knot_edit = QLineEdit()
        self.knot_edit.setReadOnly(True)
        self.knot_edit.setToolTip("NURBS/BSpline knot vectors are auto-generated to ensure compliance with specifications")
        left.addWidget(self.knot_edit)

        btn_insert_knot = QPushButton('Insert Knot (t value)')
        btn_insert_knot.clicked.connect(self.insert_knot_dialog)
        left.addWidget(btn_insert_knot)

        btn_degree_elev = QPushButton('Elevate Bézier Degree (1 order)')
        btn_degree_elev.clicked.connect(self.degree_elevate_current)
        left.addWidget(btn_degree_elev)

        btn_save = QPushButton('Save Project')
        btn_save.clicked.connect(self.save_project)
        left.addWidget(btn_save)

        btn_load = QPushButton('Load Project')
        btn_load.clicked.connect(self.load_project)
        left.addWidget(btn_load)

        btn_export_svg = QPushButton('Export SVG')
        btn_export_svg.clicked.connect(self.export_svg)
        left.addWidget(btn_export_svg)

        left.addWidget(QLabel('Fitting Degrees (comma separated):'))
        self.compare_degrees_edit = QLineEdit()
        self.compare_degrees_edit.setPlaceholderText('e.g.: 1,2,3,4')
        left.addWidget(self.compare_degrees_edit)

        left.addWidget(QLabel('Geometric Fitting Samples:'))
        self.fit_sample_spin = QSpinBox()
        self.fit_sample_spin.setRange(20, 5000)
        self.fit_sample_spin.setValue(500)
        left.addWidget(self.fit_sample_spin)

        btn_compare = QPushButton('Geometric Fitting Comparison (Optimized)')
        btn_compare.clicked.connect(self.compare_degrees_geometric)
        left.addWidget(btn_compare)

        btn_clear_compare = QPushButton('Clear Comparison Results')
        btn_clear_compare.clicked.connect(self.clear_comparisons)
        left.addWidget(btn_clear_compare)

        left.addStretch()

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.figure.tight_layout()

        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_canvas_drag)
        self.canvas.mpl_connect('button_release_event', self.on_canvas_release)

        self.dragging = False
        self.drag_index = None

        self.status = self.statusBar()
        self.status.showMessage('Ready')

        self.add_curve('bezier')

    def add_curve(self, kind='bezier'):
        c = CurveObject(kind)
        c.name = f'{kind}_{len(self.curves)}'
        default_pts = [(0.0, 0.0), (1.0, 0.5), (2.0, -0.5), (3.0, 0.0)]
        c.ctrl_pts = default_pts[:c.degree + 1]
        c.ensure_knots()
        self.curves.append(c)
        self.curve_list.addItem(c.name)
        self.curve_list.setCurrentRow(len(self.curves) - 1)
        self.redraw()

    def new_curve_dialog(self):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle('New Curve')
        v = QVBoxLayout()
        dlg.setLayout(v)
        v.addWidget(QLabel('Curve Type:'))
        cb = QComboBox()
        cb.addItems(['bezier', 'bspline', 'nurbs'])
        v.addWidget(cb)
        btn_ok = QPushButton('Create')
        btn_ok.clicked.connect(dlg.accept)
        v.addWidget(btn_ok)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.add_curve(cb.currentText())

    def delete_curve(self):
        idx = self.curve_list.currentRow()
        if idx >= 0:
            self.curves.pop(idx)
            self.curve_list.takeItem(idx)
            if len(self.curves) > 0:
                self.curve_list.setCurrentRow(0)
            else:
                self.current_curve = None
            self.redraw()

    def on_curve_selected(self, idx):
        if idx < 0 or idx >= len(self.curves):
            self.current_curve = None
            return
        self.current_curve = self.curves[idx]
        self.spin_degree.setValue(self.current_curve.degree)
        self.ctrl_index.setMaximum(max(0, len(self.current_curve.ctrl_pts) - 1))
        self.update_ctrl_fields()
        self.current_curve.ensure_knots()
        knot_text = ','.join([f'{k:.4f}' for k in self.current_curve.knots])
        self.knot_edit.setText(knot_text)
        self.redraw()

    def on_ctrl_index_changed(self, v):
        self.update_ctrl_fields()

    def update_ctrl_fields(self):
        c = self.current_curve
        if c and len(c.ctrl_pts) > 0:
            idx = self.ctrl_index.value()
            idx = min(idx, len(c.ctrl_pts) - 1)
            self.ctrl_index.setValue(idx)
            x, y = c.ctrl_pts[idx]
            self.ctrl_x.blockSignals(True)
            self.ctrl_y.blockSignals(True)
            self.ctrl_w.blockSignals(True)
            self.ctrl_x.setValue(x)
            self.ctrl_y.setValue(y)
            if c.kind == 'nurbs' and idx < len(c.weights):
                self.ctrl_w.setEnabled(True)
                self.ctrl_w.setValue(c.weights[idx])
            else:
                self.ctrl_w.setEnabled(False)
            self.ctrl_x.blockSignals(False)
            self.ctrl_y.blockSignals(False)
            self.ctrl_w.blockSignals(False)

    def on_ctrl_coord_changed(self, val):
        c = self.current_curve
        if not c or len(c.ctrl_pts) == 0:
            return
        idx = self.ctrl_index.value()
        idx = min(idx, len(c.ctrl_pts) - 1)
        c.ctrl_pts[idx] = (float(self.ctrl_x.value()), float(self.ctrl_y.value()))
        c.ensure_knots()
        self.knot_edit.setText(','.join([f'{k:.4f}' for k in c.knots]))
        self.redraw()

    def on_weight_changed(self, val):
        c = self.current_curve
        if not c or c.kind != 'nurbs':
            return
        idx = self.ctrl_index.value()
        if idx < len(c.weights):
            c.weights[idx] = float(self.ctrl_w.value())
            self.redraw()

    def on_knot_edited(self):
        QMessageBox.information(self, 'Information', 'Knot vectors are auto-generated and cannot be edited manually')
        if self.current_curve:
            self.current_curve.ensure_knots()
            self.knot_edit.setText(','.join([f'{k:.4f}' for k in self.current_curve.knots]))

    def map_to_data(self, event):
        if event.xdata is None or event.ydata is None:
            return None
        return (event.xdata, event.ydata)

    def on_canvas_click(self, event):
        if event.inaxes != self.ax:
            return
        pos = self.map_to_data(event)
        if pos is None:
            return
        x, y = pos
        if event.button == 1:
            idx, dist = self.find_nearest_ctrl(x, y)
            if dist < 0.05:
                self.dragging = True
                self.drag_index = idx
            else:
                if self.current_curve is not None:
                    self.current_curve.ctrl_pts.append((x, y))
                    if self.current_curve.kind == 'nurbs':
                        self.current_curve.weights.append(1.0)
                    self.current_curve.ensure_knots()
                    self.ctrl_index.setMaximum(max(0, len(self.current_curve.ctrl_pts) - 1))
                    self.ctrl_index.setValue(len(self.current_curve.ctrl_pts) - 1)
                    self.knot_edit.setText(','.join([f'{k:.4f}' for k in self.current_curve.knots]))
                    self.update_ctrl_fields()
                    self.redraw()
        elif event.button == 3:
            idx, dist = self.find_nearest_ctrl(x, y)
            if idx is not None and dist < 0.05:
                self.current_curve.ctrl_pts.pop(idx)
                if self.current_curve.kind == 'nurbs':
                    if idx < len(self.current_curve.weights):
                        self.current_curve.weights.pop(idx)
                self.current_curve.ensure_knots()
                self.ctrl_index.setMaximum(max(0, len(self.current_curve.ctrl_pts) - 1))
                self.ctrl_index.setValue(min(self.ctrl_index.value(), max(0, len(self.current_curve.ctrl_pts) - 1)))
                self.knot_edit.setText(','.join([f'{k:.4f}' for k in self.current_curve.knots]))
                self.update_ctrl_fields()
                self.redraw()

    def on_canvas_drag(self, event):
        if not self.dragging or self.drag_index is None or event.inaxes != self.ax:
            return
        pos = self.map_to_data(event)
        if pos is None:
            return
        x, y = pos
        c = self.current_curve
        if c and 0 <= self.drag_index < len(c.ctrl_pts):
            c.ctrl_pts[self.drag_index] = (x, y)
            self.update_ctrl_fields()
            self.redraw()

    def on_canvas_release(self, event):
        self.dragging = False
        self.drag_index = None

    def find_nearest_ctrl(self, x, y):
        if not self.current_curve or len(self.current_curve.ctrl_pts) == 0:
            return None, float('inf')
        pts = np.array(self.current_curve.ctrl_pts)
        dists = np.sqrt((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2)
        idx = int(np.argmin(dists))
        return idx, float(dists[idx])

    def insert_knot_dialog(self):
        if not self.current_curve:
            return
        if self.current_curve.kind == 'bezier':
            QMessageBox.information(self, 'Not Supported', 'Knot insertion is not supported for Bézier curves (use curve splitting instead)')
            return
        t, ok = QtWidgets.QInputDialog.getDouble(self, 'Insert Knot', 't value (0-1):', 0.5, 0.0, 1.0, 4)
        if not ok:
            return
        c = self.current_curve
        old_ctrl = c.ctrl_pts[:]
        old_w = c.weights[:] if c.kind == 'nurbs' else None
        try:
            new_ctrl, new_w, new_knots = knot_insert(old_ctrl, c.degree, c.knots, float(t), old_w)
            c.ctrl_pts = new_ctrl
            if c.kind == 'nurbs':
                c.weights = new_w
            c.knots = new_knots
            c.ensure_knots()
            self.update_ctrl_fields()
            self.knot_edit.setText(','.join([f'{k:.4f}' for k in c.knots]))
            self.redraw()
        except Exception as e:
            QMessageBox.warning(self, 'Insertion Failed', f'Knot insertion failed: {str(e)}')

    def degree_elevate_current(self):
        c = self.current_curve
        if c is None:
            return
        if c.kind != 'bezier':
            QMessageBox.information(self, 'Not Supported', 'Degree elevation is only supported for Bézier curves')
            return
        try:
            old = c.ctrl_pts[:]
            new = bezier_degree_elevate(old)
            c.ctrl_pts = [(float(p[0]), float(p[1])) for p in new]
            c.degree += 1
            c.ensure_knots()
            self.spin_degree.setValue(c.degree)
            self.ctrl_index.setMaximum(max(0, len(c.ctrl_pts) - 1))
            self.knot_edit.setText(','.join([f'{k:.4f}' for k in c.knots]))
            self.update_ctrl_fields()
            self.redraw()
        except Exception as e:
            QMessageBox.warning(self, 'Elevation Failed', str(e))

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Project', filter='JSON Files (*.json)')
        if not path:
            return
        data = [c.to_dict() for c in self.curves]
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self.status.showMessage('Saved to ' + path)

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Load Project', filter='JSON Files (*.json)')
        if not path:
            return
        with open(path, 'r') as f:
            data = json.load(f)
        self.curves = [CurveObject.from_dict(d) for d in data]
        self.curve_list.clear()
        for c in self.curves:
            self.curve_list.addItem(c.name)
        if len(self.curves) > 0:
            self.curve_list.setCurrentRow(0)
            self.current_curve = self.curves[0]
            self.knot_edit.setText(','.join([f'{k:.4f}' for k in self.current_curve.knots]))
        self.redraw()
        self.status.showMessage('Loaded ' + path)

    def export_svg(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Export SVG', filter='SVG Files (*.svg)')
        if not path:
            return
        minx, miny, maxx, maxy = None, None, None, None
        all_paths = []
        for c in self.curves:
            pts = c.sample(self.spin_sample.value())
            if pts.size == 0:
                continue
            all_paths.append(pts)
            xs = pts[:, 0]
            ys = pts[:, 1]
            if minx is None:
                minx, maxx = xs.min(), xs.max()
                miny, maxy = ys.min(), ys.max()
            else:
                minx, maxx = min(minx, xs.min()), max(maxx, xs.max())
                miny, maxy = min(miny, ys.min()), max(maxy, ys.max())
        if minx is None:
            QMessageBox.information(self, 'Empty Curves', 'No curves to export')
            return
        w = maxx - minx
        h = maxy - miny
        if w == 0: w = 1.0
        if h == 0: h = 1.0
        pad = 10
        scale = 500.0 / max(w, h)
        with open(path, 'w') as f:
            f.write('<?xml version="1.0" standalone="no"?>\n')
            f.write('<svg xmlns="http://www.w3.org/2000/svg" version="1.1">\n')
            for pts in all_paths:
                d = 'M ' + ' '.join(
                    f'{(p[0] - minx) * scale + pad:.3f} {((maxy - p[1]) * scale + pad):.3f}' for p in pts)
                f.write(f'<path d="{d}" stroke="black" fill="none" stroke-width="1" />\n')
            f.write('</svg>')
        self.status.showMessage('Exported SVG to ' + path)

    def parse_degrees_list(self, text):
        try:
            arr = [int(x.strip()) for x in text.split(',') if x.strip() != '']
            arr = [d for d in arr if d >= 0]
            return sorted(list(set(arr)))
        except Exception:
            return []

    def compare_degrees_geometric(self):
        c = self.current_curve
        if c is None:
            QMessageBox.information(self, 'No Curve', 'Please select a curve to fit')
            return

        text = self.compare_degrees_edit.text().strip()
        degrees = self.parse_degrees_list(text)
        if not degrees:
            QMessageBox.information(self, 'Invalid Degrees', 'Please enter valid fitting degrees (e.g., 1,2,3)')
            return

        fit_samples = int(self.fit_sample_spin.value())
        if fit_samples < 20:
            QMessageBox.warning(self, 'Insufficient Samples', 'Geometric fitting requires at least 20 sample points')
            return

        self.ground_truth_pts = c.sample(fit_samples)
        if len(self.ground_truth_pts) < 2:
            QMessageBox.information(self, 'Empty Curve', 'Curve has insufficient points for fitting')
            return

        results = []
        for d in degrees:
            try:
                s_param, fit_pts, rmse, coeffs_x, coeffs_y = fit_geometric_polynomial(
                    self.ground_truth_pts, d, fit_samples
                )
                results.append({
                    'degree': d,
                    's_param': s_param,
                    'fit_pts': fit_pts,
                    'rmse': rmse,
                    'coeffs_x': coeffs_x,
                    'coeffs_y': coeffs_y
                })
                print(f'Degree {d}: Geometric RMSE = {rmse:.6f}')
            except Exception as e:
                QMessageBox.warning(self, 'Fitting Failed', f'Fitting failed for degree {d}: {str(e)}')

        self.compare_results = results

        if results:
            summary = ', '.join([f'd={r["degree"]}: RMSE={r["rmse"]:.6f}' for r in results])
            self.status.showMessage(f'Geometric fitting completed: {summary}')
        else:
            self.status.showMessage('No valid fitting results')

        self.redraw()

    def clear_comparisons(self):
        self.compare_results = []
        self.ground_truth_pts = None
        self.arc_length_pts = None
        self.status.showMessage('Comparison results cleared')
        self.redraw()

    def on_degree_changed(self, v):
        c = self.current_curve
        if not c:
            return
        c.degree = int(v)
        c.ensure_knots()
        self.knot_edit.setText(','.join([f'{k:.4f}' for k in c.knots]))
        self.redraw()

    def redraw(self):
        self.ax.clear()
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.grid(True, linestyle='--', alpha=0.4)

        curve_colors = ['darkgreen', 'darkblue', 'darkred', 'darkorange', 'darkviolet', 'saddlebrown']
        fit_colors = ['red', 'blue', 'orange', 'purple', 'brown', 'pink']
        linestyles = ['-', '--', '-.', ':']

        for curve_idx, c in enumerate(self.curves):
            if len(c.ctrl_pts) == 0:
                continue
            pts = np.array(c.ctrl_pts)
            base_color = curve_colors[curve_idx % len(curve_colors)]
            self.ax.plot(pts[:, 0], pts[:, 1], marker='o', linestyle='--', linewidth=1, markersize=6,
                         label=f'{c.name} Control Points', color=base_color, alpha=0.4)
            try:
                samp = c.sample(self.spin_sample.value())
                if samp.shape[0] > 0:
                    self.ax.plot(samp[:, 0], samp[:, 1], linestyle='-', linewidth=3, alpha=0.8,
                                 label=f'{c.name} Original Curve', color=base_color)
            except Exception as e:
                print('Drawing failed:', c.name, e)

            if c == self.current_curve:
                for i, (x, y) in enumerate(c.ctrl_pts):
                    self.ax.text(x, y, str(i), fontsize=9, color='black', weight='bold')

        if self.current_curve and len(self.compare_results) > 0 and self.ground_truth_pts is not None:
            curve_idx = self.curves.index(self.current_curve)
            for fit_idx, r in enumerate(self.compare_results):
                fit_pts = r.get('fit_pts')
                if fit_pts is None or fit_pts.size == 0:
                    continue
                color = fit_colors[(curve_idx + fit_idx) % len(fit_colors)]
                style = linestyles[fit_idx % len(linestyles)]
                rmse_val = r['rmse']
                rmse_str = f'{rmse_val:.6f}' if rmse_val >= 1e-6 else f'{rmse_val:.2e}'
                self.ax.plot(
                    fit_pts[:, 0], fit_pts[:, 1],
                    linestyle=style, linewidth=2, color=color, alpha=0.7,
                    label=f'{self.current_curve.name} Fitted d={r["degree"]} (RMSE={rmse_str})'
                )

        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=8)
        self.canvas.draw()


def main():
    app = QApplication(sys.argv)
    win = CurveDesigner()
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()