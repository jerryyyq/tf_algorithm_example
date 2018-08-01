import os
from PIL import Image

'''
from img_to_tf_record import Img2TFRecord
one_Set = Img2TFRecord('/home/yangyuqi/work/crack_sample/class_small', '/home/yangyuqi/work/crack_sample/tf_record_small', 'gif')
one_Set.generate_tf_record_files( (215, 175) )
'''


INPUT_DIR   =  '/home/yangyuqi/work/crack_sample'
OUTPUT_DIR  =  '/home/yangyuqi/work/crack_sample_small'


for dir_name, dirs, files in os.walk( INPUT_DIR ):
    for file_name in files:
        dir_fields = os.path.split( dir_name )
        out_dir = os.path.join( OUTPUT_DIR, dir_fields[-1] )
        if not os.path.exists( out_dir ):
            os.makedirs( out_dir )

        old_file_path = os.path.join(dir_name, file_name)
        new_file_path = os.path.join(out_dir, file_name)

        img = Image.open( old_file_path )
        resize_img = img.resize((int(img.width/2), int(img.height/2)), Image.ANTIALIAS)
        resize_img.save(new_file_path)

