# import network2
import webcam
# import asyncio
# import preprocess
# import augment
# import analyzer
# import data
# import helpers as hlp
import serializer
import network2

if __name__ == '__main__':
    # serializer.Serializer.load_npy_datasets()
    # webcam.launch_webcam()
    # webcam.async_webcam()
    network2.run_network(enable_gpu=False, enable_hyperopt=False)
    # preprocess.get_data(split_data=False, include_ck=True)
    # preprocess.get_data(split_data=True)
    # network2.run_network(enable_gpu=True, enable_hyperopt=True)
