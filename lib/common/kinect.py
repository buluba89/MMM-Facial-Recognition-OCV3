import threading
import time
import cv2
from pylibfreenect2 import Freenect2, SyncMultiFrameListener
from pylibfreenect2 import FrameType, setGlobalLogger, createConsoleLogger, LoggerLevel
try:
    from pylibfreenect2 import OpenGLPacketPipeline
    pipeline = OpenGLPacketPipeline()
except:
    try:
        from pylibfreenect2 import OpenCLPacketPipeline
        pipeline = OpenCLPacketPipeline()
    except:
        from pylibfreenect2 import CpuPacketPipeline
        pipeline = CpuPacketPipeline()

CAPTURE_HZ = 30.0


class OpenCVCapture(object):
    def __init__(self, device_id=0):
        """Create an OpenCV capture object associated with the provided webcam
        device ID.
        """
        logger = createConsoleLogger(LoggerLevel.Error)
        setGlobalLogger(logger)
        self.fn = Freenect2()
        num_devices = self.fn.enumerateDevices()
        if num_devices == 0:
            print("No Kinect!")
            raise LookupError()

        serial = self.fn.getDeviceSerialNumber(0)
        self.device = self.fn.openDevice(serial, pipeline=pipeline)

        types = 0
        types |= FrameType.Color

        self.listener = SyncMultiFrameListener(types)
        self.device.setColorFrameListener(self.listener)

        # Start a thread to continuously capture frames.
        # This must be done because different layers of buffering in the webcam
        # and OS drivers will cause you to retrieve old frames if they aren't
        # continuously read.
        self._capture_frame = None
        # Use a lock to prevent access concurrent access to the camera.
        self._capture_lock = threading.Lock()
        self._capture_thread = threading.Thread(target=self._grab_frames)
        self._capture_thread.daemon = True
        self._capture_thread.start()

    def _grab_frames(self):
        self.device.startStreams(rgb=True, depth=False)

        while True:
            frames = self.listener.waitForNewFrame()
            color = frames["color"]

            with self._capture_lock:
                self._capture_frame = None
                self._capture_frame =  cv2.resize(color.asarray(), (int(1920 / 3), int(1080 / 3))).copy()

            self.listener.release(frames)
            time.sleep(1.0 / CAPTURE_HZ)

    def read(self):
        """Read a single frame from the camera and return the data as an OpenCV
        image (which is a numpy array).
        """
        frame = None
        with self._capture_lock:
            frame = self._capture_frame
        # If there are problems, keep retrying until an image can be read.
        while frame is None:
            time.sleep(0)
            with self._capture_lock:
                frame = self._capture_frame
        # Return the capture image data.
        return frame

    def stop(self):
        print('{"status":"Terminating..."}')
        self.device.stop()
        self.device.close()
