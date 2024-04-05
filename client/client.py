import asyncio
from aiortc import RTCPeerConnection
from aiortc.contrib.signaling import TcpSocketSignaling
import cv2
import numpy as np
import multiprocessing
import json
import ctypes

def find_ball_coordinates(frame):
    """
    Detects the ball in the frame and returns its coordinates.
    
    Args:
        frame (np.array): The current video frame.
        
    Returns:
        tuple: The x and y coordinates of the ball, or None if not found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        (x, y), _ = cv2.minEnclosingCircle(c)
        return int(x), int(y)
    return None

def process_frame(frame_queue, x_value, y_value):
    """
    Processes each frame from the queue to detect the ball and updates shared x and y coordinates.
    
    Args:
        frame_queue (multiprocessing.Queue): Queue containing frames to be processed.
        x_value (multiprocessing.Value): Shared object for the ball's x-coordinate.
        y_value (multiprocessing.Value): Shared object for the ball's y-coordinate.
    """
    while True:
        frame = frame_queue.get()
        if frame is None:  # Check for shutdown signal.
            break
        coordinates = find_ball_coordinates(frame)
        if coordinates:
            x_value.value, y_value.value = coordinates

async def send_coordinates(channel, x_value, y_value):
    """
    Periodically sends the detected ball coordinates over the data channel to the server.
    
    Args:
        channel: The WebRTC data channel for sending coordinates.
        x_value (multiprocessing.Value): Shared object for the ball's x-coordinate.
        y_value (multiprocessing.Value): Shared object for the ball's y-coordinate.
    """
    while True:
        await asyncio.sleep(0.1)  # Send rate can be adjusted.
        if channel.readyState == "open":
            coordinates = json.dumps({"x": x_value.value, "y": y_value.value})
            print(f"Sending coordinates: {coordinates}")
            channel.send(coordinates)

async def run_client():
    """
    Initializes processing and sets up communication with the server.
    """
    frame_queue = multiprocessing.Queue()
    x_value = multiprocessing.Value(ctypes.c_int, 0)
    y_value = multiprocessing.Value(ctypes.c_int, 0)
    process_a = multiprocessing.Process(target=process_frame, args=(frame_queue, x_value, y_value))
    process_a.start()

    signaling = TcpSocketSignaling("127.0.0.1", 9000)
    # replace the above line by the bellow line in case you want to use docker.
    # signaling = TcpSocketSignaling("server", 9000)


    await signaling.connect()
    pc = RTCPeerConnection()

    @pc.on("datachannel")
    def on_datachannel(channel):
        print(f"Data channel received: {channel.label}")
        asyncio.create_task(send_coordinates(channel, x_value, y_value))

    @pc.on("track")
    async def on_track(track):
        print("First Track received")
        while True:
            frame = await track.recv()
            img = frame.to_ndarray(format="bgr24")
            frame_queue.put(img)  # Send the frame to be processed.

            if (x_value.value, y_value.value) != (0, 0):  # Check if coordinates have been updated.
                cv2.circle(img, (x_value.value, y_value.value), 5, (0, 0, 255), -1)
                cv2.imshow("Bouncing Ball", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    offer = await signaling.receive()
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    await signaling.send(pc.localDescription)

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        frame_queue.put(None)  # Shutdown signal for the processing process.
        process_a.join()
        await pc.close()
        await signaling.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(run_client())
