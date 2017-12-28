# import network2
# import webcam
# import asyncio
import preprocess

if __name__ == '__main__':
    # webcam.launch_webcam()
    # webcam.async_webcam()
    # network2.run_network()
    # preprocess.get_data(split_data=False, include_ck=True)
    # preprocess.get_data(split_data=True)
    features, labels = preprocess.load_from_npy(split_data=True, shuffle_data=False)
