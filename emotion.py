# import network2
# import webcam
# import asyncio
import preprocess
import augment
import analyzer
import data
import helpers as hlp

if __name__ == '__main__':
    # webcam.launch_webcam()
    # webcam.async_webcam()
    # network2.run_network()
    # preprocess.get_data(split_data=False, include_ck=True)
    # preprocess.get_data(split_data=True)
    features, labels = preprocess.load_from_npy(split_data=False, shuffle_data=False)
    print(analyzer.label_cnt(labels))
    features, labels = augment.augment_data(features, labels)
    print(analyzer.label_cnt(labels))
    emotion_groups = preprocess.split_emotions(features, labels)
    set_distribution = [0.94, 0.03, 0.03]

    # test_set = dataset[<emotion id>][<feature>/<label>][<set>] ->
    # dataset[<set>][<feature>/<label>]
    dataset = [(preprocess.perc_split(group[0], set_distribution), preprocess.perc_split(group[1], set_distribution))
               for
               group in emotion_groups]

    datasets = [(hlp.merge([dataset[emotion_id][0][set_id] for emotion_id in range(data.N_CLASSES)]),
                 hlp.merge([dataset[emotion_id][1][set_id] for emotion_id in range(data.N_CLASSES)])) for set_id in
                range(3)]

    print(dataset)
