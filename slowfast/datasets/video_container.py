#!/usr/bin/env python3

import av
import decord
from decord import VideoReader
from decord import cpu
import os
def get_video_container(path_to_vid, multi_thread_decode=False, backend="pyav"):
    """
    Given the path to the video, return the pyav video container.
    Args:
        path_to_vid (str): path to the video.
        multi_thread_decode (bool): if True, perform multi-thread decoding.
        backend (str): decoder backend, options include `pyav` and
            `torchvision`, default is `pyav`.
    Returns:
        container (container): video container.
    """
    if backend == "torchvision":
        if client:
            video_bytes = client.get(path_to_vid)
            container = memoryview(video_bytes)
        else:
            with open(path_to_vid, "rb") as fp:
                container = fp.read()
        return container
    elif backend == "pyav":
        container = av.open(path_to_vid)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        return container
    elif backend == "decord":
        # print(path_to_vid)
        # commd="ffmpeg -r 10 -f image2 -i {}/%d.png {}../output1.mp4 -y  -loglevel quiet".format(path_to_vid,path_to_vid)
        
        # pp=os.path.join(path_to_vid,"../output1.mp4")
        if not path_to_vid.endswith(".mp4"):
            path_to_vid=path_to_vid+"/output1.mp4"
        
        container = VideoReader(path_to_vid, ctx=cpu(0))
        decord.bridge.set_bridge('torch')
        # print(container)
        # print(type(container))
        return container
    else:
        raise NotImplementedError("Unknown backend {}".format(backend))
if __name__ =="__main__":
    path=""