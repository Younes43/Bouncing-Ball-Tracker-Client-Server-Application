import numpy as np
import cv2
import pytest
from client import find_ball_coordinates, process_frame 
from multiprocessing import Process, Queue, Value
import ctypes

def create_test_frame(ball_positions, frame_size=(480, 640), ball_radius=30):
    """
    Creates a test frame with balls at specified positions.

    Args:
        ball_positions (list): A list of (x, y) coordinates for balls.
        frame_size (tuple, optional): The size of the frame.
        ball_radius (int, optional): The radius of each ball.

    Returns:
        np.array: The generated frame.
    """
    frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
    for position in ball_positions:
        cv2.circle(frame, position, ball_radius, (255, 255, 255), -1)
    return frame

def test_single_ball_center():
    """Tests detection of a single ball located in the center of the frame."""
    frame = create_test_frame([(320, 240)])
    assert find_ball_coordinates(frame) == (320, 240)

def test_multiple_balls():
    """Tests detection of the largest ball when multiple balls are present."""
    frame = create_test_frame([(100, 100), (200, 200)], ball_radius=20)
    assert find_ball_coordinates(frame) == (200, 200), "The function should identify the largest ball."

def test_no_balls():
    """Tests that None is returned when no balls are present in the frame."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert find_ball_coordinates(frame) is None, "Expected None when no balls are present."

def test_ball_on_edge():
    """Tests detection of a ball located at the edge of the frame."""
    frame = create_test_frame([(639, 1)], ball_radius=30)
    assert find_ball_coordinates(frame) is not None, "Ball on the edge should still be detected."

def test_ball_partially_out_of_frame():
    """Tests detection of a ball that is partially outside the frame."""
    frame = create_test_frame([(630, 30)], ball_radius=30)
    assert find_ball_coordinates(frame) is not None, "Partially visible ball should still be detected."

@pytest.fixture
def setup_environment():
    """
    Sets up the environment for testing process_frame, including a frame queue and shared x, y values.
    """
    frame_queue = Queue()
    x_value = Value(ctypes.c_int, 0)
    y_value = Value(ctypes.c_int, 0)
    return frame_queue, x_value, y_value

def test_process_frame_with_ball(setup_environment):
    """
    Tests that process_frame correctly updates shared x and y values when a ball is present in the frame.
    """
    frame_queue, x_value, y_value = setup_environment
    # Assuming create_test_frame is defined to create frames with specified ball positions
    test_frame = create_test_frame([(320, 240)])  # Place a ball in the center

    # Put the test frame in the queue and a None to signal the end of the queue
    frame_queue.put(test_frame)
    frame_queue.put(None)

    # Start the process_frame function in a separate process
    process = Process(target=process_frame, args=(frame_queue, x_value, y_value))
    process.start()
    process.join()

    assert x_value.value == 320 and y_value.value == 240, "Failed to detect and update ball coordinates correctly."

def test_process_frame_without_ball(setup_environment):
    """
    Tests that process_frame does not update shared x and y values when no ball is present in the frame.
    """
    frame_queue, x_value, y_value = setup_environment
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # An empty frame without a ball

    frame_queue.put(empty_frame)
    frame_queue.put(None)  # Signal the end

    process = Process(target=process_frame, args=(frame_queue, x_value, y_value))
    process.start()
    process.join()

    # Assuming the initial values of x_value and y_value are 0
    assert x_value.value == 0 and y_value.value == 0, "Incorrectly updated coordinates for a frame without a ball."

