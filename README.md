# Curve_Design_Tool
Implement an interactive curve modeling system that supports the creation, editing, geometric operations, and export of Bézier and B-spline/NURBS curves. It can visualize curves, curvature, and continuity (G0/G1/G2) and demonstrate several algorithms (De Casteljau, knot insertion, degree elevation, etc.).
It supports at least two types of curve representations: Bézier (arbitrary order) and NURBS (arbitrary order, multi-node).

Interactive editing: drag and drop control points, add/delete control points, move weights, modify node vectors, and switch orders (degree elevation/reduction optional).

Real-time visualization: curves, control polygons, parametric points, radii of curvature, tangents, curvature plots (kappa(u)) and error measurements.

Implement and compare the core algorithms: De Casteljau, Cox–de Boor (B-spline evaluation), knot insertion, degree elevation, and curve splitting.

It can import/export common formats (such as JSON, SVG (2D), OBJ (dots and lines only) or IGES/STEP (for advanced users)).

Final submissions: source code, executable demo, user manual, and experiment report (including algorithm complexity/numerical stability analysis and several use cases).
