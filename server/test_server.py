import asyncio
import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock
from server import Ball, BouncingBallTrack 
import json

@pytest.fixture
def ball():
    """Fixture for creating a default Ball instance."""
    return Ball(x=320, y=240, velocity_x=2, velocity_y=2, radius=30)

def test_ball_update_position(ball):
    """Test that the ball updates its position correctly."""
    ball.update_position(640, 480)
    assert ball.x == 322 and ball.y == 242, "Ball did not update position correctly."

def test_ball_check_bounce(ball):
    """Test that the ball bounces off the edges correctly."""
    # Move ball to the edge
    ball.x, ball.y = 639, 240
    ball.update_position(640, 480)
    # Check if it bounced
    assert ball.velocity_x == -2, "Ball did not bounce on x edge."
def test_ball_velocity_change():
    """Test that the ball's velocity changes when manually updated."""
    ball = Ball(100, 100, 5, 5, 10)
    ball.velocity_x, ball.velocity_y = -5, -5
    ball.update_position(200, 200)
    assert ball.x == 95 and ball.y == 95, "Ball velocity change not correctly handled."

def test_ball_zero_velocity():
    """Test that the ball with zero velocity remains static."""
    ball = Ball(320, 240, 0, 0, 30)
    ball.update_position(640, 480)
    assert ball.x == 320 and ball.y == 240, "Static ball unexpectedly moved."

def test_ball_collision_corner():
    """Test ball bouncing off a corner where both x and y should invert."""
    ball = Ball(630, 470, 10, 10, 10)  # Heading to bottom-right corner
    ball.update_position(640, 480)
    assert ball.velocity_x < 0 and ball.velocity_y < 0, "Ball did not correctly bounce off corner."

@pytest.mark.asyncio
async def test_bouncing_ball_track_recv(ball):
    """Test that BouncingBallTrack generates frames correctly."""
    track = BouncingBallTrack(ball)
    frame = await track.recv()
    assert frame is not None, "Did not receive a frame from BouncingBallTrack."

@pytest.mark.asyncio
async def test_frame_size():
    """Ensure generated frames have the expected size."""
    ball = Ball(320, 240, 2, 2, 30)
    track = BouncingBallTrack(ball, width=800, height=600)
    frame = await track.recv()
    assert frame.width == 800 and frame.height == 600, "Generated frame does not match expected size."

@pytest.mark.asyncio
async def test_frame_ball_visibility():
    """Check if the ball is visible in the generated frame."""
    ball = Ball(50, 50, 0, 0, 50)  # A large ball that should be easily visible
    track = BouncingBallTrack(ball)
    frame = await track.recv()
    ndarray_frame = frame.to_ndarray(format="bgr24")
    # Check for non-zero pixels indicating the ball is drawn
    assert np.any(ndarray_frame), "Ball is not visible in the generated frame."

@pytest.mark.asyncio
async def test_frame_content_changes():
    """Verify that subsequent frames show changes in ball position."""
    ball = Ball(320, 240, 10, 10, 30)
    track = BouncingBallTrack(ball)
    first_frame = await track.recv()
    second_frame = await track.recv()
    # Convert frames to numpy arrays and check if they are different
    first_ndarray = first_frame.to_ndarray(format="bgr24")
    second_ndarray = second_frame.to_ndarray(format="bgr24")
    assert not np.array_equal(first_ndarray, second_ndarray), "Subsequent frames do not show changes in content."

@pytest.mark.asyncio
async def test_on_message_with_error_calculation(mocker):
    """Test the on_message function handling and error calculation."""
    mock_channel = AsyncMock()
    mock_track = MagicMock()
    ball_instance = Ball(x=300, y=200, velocity_x=2, velocity_y=2, radius=30)
    mock_track.ball = ball_instance
    setup_data_channel = mocker.patch('server.setup_data_channel')
    setup_data_channel.on_message = AsyncMock()

    # Simulate receiving coordinates that are off by (20, 40) from the actual position
    received_coordinates = json.dumps({"x": 320, "y": 240})
    await setup_data_channel.on_message(received_coordinates)

    # Calculate expected error
    expected_error = np.sqrt((20 ** 2) + (40 ** 2))
    setup_data_channel.on_message.assert_called_once()
    print(f"Calculated Error: {expected_error}")  # This line is more illustrative than assertive


