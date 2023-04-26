import numpy as np
import gradio as gr
from utils.swap_func import run_inference
import cv2
from tensorflow.keras.models import load_model
from retinaface.models import *
from utils.utils import (estimate_norm, get_lm, inverse_estimate_norm,
                         norm_crop, transform_landmark_points)
from scipy.ndimage import gaussian_filter
from tensorflow_addons.layers import InstanceNormalization

from networks.layers import AdaIN, AdaptiveAttention
import sys
from moviepy.editor import AudioFileClip, VideoFileClip
import os
from tqdm import tqdm
import shutil
import proglog
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import subprocess
import glob
RetinaFace = load_model("./retinaface/RetinaFace-Res50.h5", compile=False,
                            custom_objects={"FPN": FPN,
                                            "SSH": SSH,
                                            "BboxHead": BboxHead,
                                            "LandmarkHead": LandmarkHead,
                                            "ClassHead": ClassHead})
ArcFace = load_model("./arcface_model/arcface/ArcFace-Res50.h5", compile=False)
Generator = load_model("./exports/myfacedancer/facedancer_30.h5", compile=False,
                   custom_objects={"AdaIN": AdaIN,
                                   "AdaptiveAttention": AdaptiveAttention,
                                   "InstanceNormalization": InstanceNormalization})
def swap_face(source,target,result_img_path="tmp.png",video=False,last_image=1):
    try:
        target = np.array(target)
        
        source_z=None
        if source_z is None:
            #source = cv2.imread(source)
            
            source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
            source = np.array(source)

            source_h, source_w, _ = source.shape #height,width
            source_a = RetinaFace(np.expand_dims(source, axis=0)).numpy()[0] #landmarks
            source_lm = get_lm(source_a, source_w, source_h)
            #align faces
            source_aligned = norm_crop(source, source_lm, image_size=112, shrink_factor=1.0)
        

            source_z = ArcFace.predict(np.expand_dims(source_aligned / 255.0, axis=0))
        #print("source_z:",source_z)
        blend_mask_base = np.zeros(shape=(256, 256, 1))
        blend_mask_base[77:240, 32:224] = 1
        blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)

        im = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
        im_h, im_w, _ = im.shape
        im_shape = (im_w, im_h)

        detection_scale = (im_w // 640) if (im_w > 640) else 1
        faces = RetinaFace(np.expand_dims(cv2.resize(im,
                                                     (im_w // detection_scale,
                                                      im_h // detection_scale)), axis=0)).numpy()
        total_img = im / 255.0

        for annotation in faces:
            lm_align = get_lm(annotation, im_w, im_h)

            # align the detected face
            M, pose_index = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
            im_aligned = cv2.warpAffine(im, M, (256, 256), borderValue=0.0)
         

            # face swap
        
            face_swap = Generator.predict([np.expand_dims((im_aligned - 127.5) / 127.5, axis=0), source_z])
            face_swap = (face_swap[0] + 1) / 2

            # get inverse transformation landmarks
            transformed_lmk = transform_landmark_points(M, lm_align)

            # warp image back
            iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
            iim_aligned = cv2.warpAffine(face_swap, iM, im_shape, borderValue=0.0)

            # blend swapped face with target image
            blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
            blend_mask = np.expand_dims(blend_mask, axis=-1)

            total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))


        total_img = np.clip(total_img * 255, 0, 255).astype('uint8')
        total_img=cv2.cvtColor(total_img, cv2.COLOR_BGR2RGB)
        if video==True:
             total_img=cv2.addWeighted(total_img,0.5,last_image,0.5,0)
        cv2.imwrite(result_img_path, total_img)
        return total_img

    except Exception as e:
        print('\n', e)
        sys.exit(0)
    

def swap_video(source,target):
   
    input_video=target
    video_forcheck = VideoFileClip(input_video)

    if video_forcheck.audio is None:
        no_audio = True
    else:
        no_audio = False

    del video_forcheck

    if not no_audio:
        video_audio_clip = AudioFileClip(input_video)

    video = cv2.VideoCapture(input_video)
    ret = True
    frame_index = 0
    temp_results_dir = './tmp_frames'

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    if os.path.exists(temp_results_dir):
        shutil.rmtree(temp_results_dir)
    os.makedirs(temp_results_dir, exist_ok=True)

    source_z = None
    n,last_image=video.read()
    for frame_index in tqdm(range(frame_count-1)):
        ret, frame = video.read()
        if ret:
            last_image = swap_face(source, frame,
                                        os.path.join('./tmp_frames', 'frame_{:0>7d}.png'.format(frame_index)),video=True,last_image=last_image
                                        )
    video.release()

    path = os.path.join('./tmp_frames', '*.png')
    image_filenames = sorted(glob.glob(path))
    clips = ImageSequenceClip(image_filenames, fps=fps)
    #name = os.path.splitext(out_video_filename)[0]

    if not no_audio:
        clips = clips.set_audio(video_audio_clip)

    out_video_filename="./swap_video.mp4"

    try:
        clips.write_videofile(out_video_filename, codec='libx264', audio_codec='aac', ffmpeg_params=[
                '-pix_fmt:v', 'yuv420p', '-colorspace:v', 'bt709', '-color_primaries:v', 'bt709',
                '-color_trc:v', 'bt709', '-color_range:v', 'tv', '-movflags', '+faststart'],
                                  logger=proglog.TqdmProgressBarLogger(print_messages=False))
    except Exception as e:
        print("\nERROR! Failed to export video")
        print('\n', e)
        sys.exit(0)

    return os.path.join(os.path.dirname(__file__),out_video_filename)
    


with gr.Blocks() as demo:
    gr.Markdown(
    """
    # Face Swap
    
    """)
    with gr.Tab("Swap Image"):
        with gr.Row():
                    with gr.Column():
                         
                        source = gr.Image(label="source")
                        target =gr.Image(label="target")
                    image_output = gr.Image()
        image_button = gr.Button("Swap Faces")
        image_button.click(swap_face, inputs=[source,target], outputs=image_output)
    with gr.Tab("Swap Video"):
        with gr.Row():
                    with gr.Column():
                         
                        source = gr.Image(label="source")
                        target =gr.Video(label="target video")
                    image_output = gr.Video()
        image_button = gr.Button("Swap Faces")
        image_button.click(swap_video, inputs=[source,target], outputs=image_output)



demo.launch()