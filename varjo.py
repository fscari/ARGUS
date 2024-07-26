import os
import ctypes
import time
import numpy as np
from multiprocessing import Process, Manager

class Matrix4x4(ctypes.Structure):
    _fields_ = [
        ('value', ctypes.c_double * 16),
    ]

class varjo_Ray(ctypes.Structure):
    _fields_ = [
        ('forward', ctypes.c_double * 3),
        ('origin', ctypes.c_double * 3),
    ]

class varjo_Gaze(ctypes.Structure):
    _fields_ = [
        ('captureTime', ctypes.c_int64),
        ('focusDistance', ctypes.c_double),
        ('frameNumber', ctypes.c_int64),
        ('gaze', varjo_Ray),
        ('leftEye', varjo_Ray),
        ('leftPupilSize', ctypes.c_double),
        ('leftStatus', ctypes.c_int64),
        ('rightEye', varjo_Ray),
        ('rightPupilSize', ctypes.c_double),
        ('rightStatus', ctypes.c_int64),
        ('stability', ctypes.c_double),
        ('status', ctypes.c_int64),
    ]

def varjo_yaw_data(shared_dict):
    try:
        # Import dll and define return types for all functions
        _dll_handle = ctypes.windll.LoadLibrary(
            os.path.join('C:\\', 'Users', 'localadmin', 'Desktop', 'varjo-sdk', 'bin', 'VarjoLib.dll'))

        _dll_handle.varjo_FrameGetPose.restype = Matrix4x4
        _dll_handle.varjo_GetCurrentTime.restype = ctypes.c_uint64
        _dll_handle.varjo_IsAvailable.restype = ctypes.c_bool
        _dll_handle.varjo_GetErrorDesc.restype = ctypes.POINTER(
            ctypes.c_char * 50)
        _dll_handle.varjo_GetError.restype = ctypes.c_int64
        _dll_handle.varjo_GetViewCount.restype = ctypes.c_int32
        _dll_handle.varjo_SessionInit.restype = ctypes.POINTER(ctypes.c_void_p)
        _dll_handle.varjo_FrameGetDisplayTime.restype = ctypes.c_int64
        _dll_handle.varjo_GetCurrentTime.restype = ctypes.c_int64

        # Initialize running session on Varjo base
        varjo_session_pointer = _dll_handle.varjo_SessionInit()

        # Initialize Pointer
        _dll_handle.varjo_CreateFrameInfo.restype = ctypes.POINTER(varjo_Gaze)
        varjo_gaze_pointer = _dll_handle.varjo_CreateFrameInfo(varjo_session_pointer)

        while True:
            try:
                _dll_handle.varjo_WaitSync(varjo_session_pointer, varjo_gaze_pointer)
                matrix = _dll_handle.varjo_FrameGetPose(varjo_session_pointer, ctypes.c_int64(2))
                matrix = list(matrix.value)
                pose_matrix = np.array(matrix).reshape((4, 4))
                pose_matrix_t = pose_matrix.transpose()
                rotation_matrix = pose_matrix_t[:3, :3]
                beta = np.degrees(-np.arcsin(rotation_matrix[2,0]))
                HMD_rotation_yaw = -beta
                shared_dict['yaw'] = HMD_rotation_yaw
                time.sleep(0.1)  # Adjust the delay as needed
            except Exception as e:
                print(f"Error in yaw data loop: {e}")
                break

    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    manager = Manager()
    shared_dict = manager.dict()

    p = Process(target=varjo_yaw_data, args=(shared_dict,))
    p.start()

    while True:
        print(shared_dict.get('yaw', 'No data'))
        time.sleep(1)
