# AirCanvas

# Virtual Paint App with Hand Gesture Recognition

This is a computer vision project that lets you draw on a virtual canvas using your hand gestures, powered by OpenCV and Mediapipe. You can switch between **Draw** mode and **Shapes** mode, allowing you to draw freehand lines or interactively select, drag, and fix shapes (rectangles, circles, arrows).

---

## 🎨 Features

- **Freehand Drawing**: Draw smooth lines on the virtual canvas using your index finger.
- **Shape Manipulation**: Select, drag, and fix predefined shapes such as:
  - Rectangle
  - Circle
  - Rectangle (different style)
  - Arrow
- **Mode Switching**: Toggle between Draw mode and Shapes mode via on-screen buttons.
- **Canvas Management**:
  - Clear the canvas using the "CLEAR" button.
  - Reset shapes or lines anytime using the **ESC** key.
- **Hand Tracking**: Uses Mediapipe’s hand landmarks to detect finger positions and gestures.

---

## 🛠️ Dependencies

- Python 3.x
- OpenCV (`cv2`)
- Mediapipe (`mediapipe`)
- NumPy (`numpy`)

You can install them via:

```bash
pip install opencv-python mediapipe numpy
