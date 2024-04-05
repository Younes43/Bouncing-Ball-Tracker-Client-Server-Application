import asyncio
import cv2
import numpy as np
from aiortc import RTCPeerConnection, VideoStreamTrack
from aiortc.contrib.signaling import TcpSocketSignaling
from av import VideoFrame
from fractions import Fraction
import json
import math

class Ball:
    """
    Represents the bouncing ball with position and velocity.
    
    Attributes:
        x (int): The x-coordinate of the ball's position.
        y (int): The y-coordinate of the ball's position.
        velocity_x (int): The velocity of the ball along the x-axis.
        velocity_y (int): The velocity of the ball along the y-axis.
        radius (int): The radius of the ball.
    """
    def __init__(self, x, y, velocity_x, velocity_y, radius):
        """
        Initializes the Ball with its position, velocity, and radius.
        """
        self.x = x
        self.y = y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.radius = radius

    def update_position(self, width, height):
        """
        Updates the ball's position based on its velocity and bounces off edges.

        Args:
            width (int): The width of the frame.
            height (int): The height of the frame.
        """
        self.x += self.velocity_x
        self.y += self.velocity_y
        self._check_bounce(width, height)

    def _check_bounce(self, width, height):
        """
        Checks and reverses the velocity if the ball hits the frame's edge.
        """
        if self.x - self.radius <= 0 or self.x + self.radius >= width:
            self.velocity_x *= -1
        if self.y - self.radius <= 0 or self.y + self.radius >= height:
            self.velocity_y *= -1

class BouncingBallTrack(VideoStreamTrack):
    """
    A video stream track that simulates a bouncing ball.

    Attributes:
        ball (Ball): The Ball object that this track visualizes.
        width (int): The width of the video frame.
        height (int): The height of the video frame.
    """
    def __init__(self, ball, width=640, height=480):
        """
        Initializes the BouncingBallTrack with a ball and frame dimensions.
        """
        super().__init__()
        self.ball = ball
        self.width = width
        self.height = height
        self.time_base = Fraction(1, 90000)
        self.pts = 0

    async def recv(self):
        """
        Generates and returns a video frame with the current ball position.
        """
        frame = np.zeros((self.height, self.width, 3), np.uint8)
        self.ball.update_position(self.width, self.height)
        cv2.circle(frame, (self.ball.x, self.ball.y), self.ball.radius, (255, 255, 255), -1)
        return self._create_video_frame(frame)

    def _create_video_frame(self, frame):
        """
        Converts a numpy array to a VideoFrame.

        Args:
            frame (np.array): The current frame as a numpy array.

        Returns:
            VideoFrame: The frame converted to a VideoFrame.
        """
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        new_frame.pts = self.pts
        new_frame.time_base = self.time_base
        self.pts += self.time_base.denominator // 30  # Assuming 30 fps
        return new_frame

async def run_server(signaling):
    """
    Sets up and runs the WebRTC server.

    Args:
        signaling (TcpSocketSignaling): The signaling mechanism for WebRTC connection.
    """
    pc = RTCPeerConnection()
    ball = Ball(x=320, y=240, velocity_x=2, velocity_y=2, radius=30)
    track = BouncingBallTrack(ball)

    pc.addTrack(track)
    await setup_data_channel(pc, track)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await signaling.send(pc.localDescription)

    answer = await signaling.receive()
    await pc.setRemoteDescription(answer)

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        await pc.close()

async def setup_data_channel(pc, track):
    """
    Configures the data channel for receiving coordinates and computes errors.

    Args:
        pc (RTCPeerConnection): The peer connection object.
        track (BouncingBallTrack): The video track that simulates the bouncing ball.
    """
    coordinate_channel = pc.createDataChannel("coordinates")

    @coordinate_channel.on("open")
    def on_open():
        print("Coordinate Data Channel opened by the server.")

    @coordinate_channel.on("message")
    def on_message(message):
        """
        Processes received coordinates and calculates the error from the actual position.

        Args:
            message (str): The received message in JSON format containing predicted coordinates.
        """
        coordinates = json.loads(message)
        predicted_x, predicted_y = coordinates["x"], coordinates["y"]
        actual_x, actual_y = track.ball.x, track.ball.y
        error = math.sqrt((actual_x - predicted_x) ** 2 + (actual_y - predicted_y) ** 2)
        print(f"Received Coordinates: {coordinates}, Actual: ({actual_x}, {actual_y}), Error: {error:.2f}")

if __name__ == "__main__":
    signaling = TcpSocketSignaling("127.0.0.1", 9000)
    # replace the above line by the bellow line in case you want to use docker.
    # signaling = TcpSocketSignaling("server", 9000)

    asyncio.run(run_server(signaling))
