import cv2
import dlib
import numpy as np
import vtk

cap = cv2.VideoCapture(0)  # Use the first webcam

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([(p.x, p.y) for p in landmarks.parts()])
        
        # You can visualize these points on the frame
        for (x, y) in points:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

def simulate_3d_effect(points):
    z_factor = np.array([1.0] * 17 + [0.85] * 10 + [0.9] * 5 + [1.4] * 5 + [1.2] * 12 + [1.1] * 12 + [1.3] * 7)
    points_3d = np.zeros((points.shape[0], 3))
    points_3d[:, :2] = points
    points_3d[:, 1] *= z_factor
    return points_3d

def create_vtk_renderer():
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    return renderer, render_window, render_window_interactor

def render_3d_face(points_3d, renderer):
    vtk_points = vtk.vtkPoints()
    for point in points_3d:
        vtk_points.InsertNextPoint(point)
    
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    
    vertex_glyph_filter = vtk.vtkVertexGlyphFilter()
    vertex_glyph_filter.SetInputData(poly_data)
    vertex_glyph_filter.Update()
    
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vertex_glyph_filter.GetOutputPort())
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    renderer.AddActor(actor)
    renderer.ResetCamera()

renderer, render_window, render_window_interactor = create_vtk_renderer()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        points = np.array([(p.x, p.y) for p in landmarks.parts()])
        points_3d = simulate_3d_effect(points)
        render_3d_face(points_3d, renderer)

    render_window.Render()
    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
render_window_interactor.Start()
