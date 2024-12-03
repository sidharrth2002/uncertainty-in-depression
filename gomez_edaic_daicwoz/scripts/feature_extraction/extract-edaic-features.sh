#!/bin/bash

set -e

NO_CHUNKED_DIR=/home/sn666/processed_data/E-DAIC/no-chunked/

# python ./scripts/feature_extraction/edaic/untar_data.py --root-dir /home/sn666/data/ --dest-dir /home/sn666/processed_data/E-DAIC/original_data/
# python ./scripts/feature_extraction/edaic/get_no_idxs.py --data-dir /home/sn666/processed_data/E-DAIC/original_data/ --dest-dir /home/sn666/processed_data/E-DAIC/no-chunked/

# # MFCC
# python ./scripts/feature_extraction/edaic/prepare_mfcc.py --src-root /home/sn666/processed_data/E-DAIC/original_data/ --modality-id audio_mfcc --dest-root $NO_CHUNKED_DIR
# python ./scripts/feature_extraction/edaic/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id audio_mfcc --no-idxs-id no_voice_idxs --dest-dir /home/sn666/processed_data/E-DAIC/data/ --nseconds 60 --frame-rate 25

# # eGeMaps
python ./scripts/feature_extraction/edaic/prepare_egemaps.py --src-root /home/sn666/processed_data/E-DAIC/original_data/ --modality-id audio_egemaps --dest-root $NO_CHUNKED_DIR
python ./scripts/feature_extraction/edaic/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id audio_egemaps --no-idxs-id no_voice_idxs --dest-dir /home/sn666/processed_data/E-DAIC/data/ --nseconds 60 --frame-rate 25

# # cnn face features
# python ./scripts/feature_extraction/edaic/prepare_cnn_resnet.py --src-root ./processed_data/E-DAIC/original_data/ --modality-id video_cnn_resnet --dest-root $NO_CHUNKED_DIR
# python ./scripts/feature_extraction/edaic/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id video_cnn_resnet --no-idxs-id no_face_idxs --dest-dir ./processed_data/E-DAIC/data/

# # gaze pose aus
# python ./scripts/feature_extraction/edaic/prepare_pose_gaze_aus.py --src-root /home/sn666/processed_data/E-DAIC/original_data/ --modality_id video_pose_gaze_aus --dest-root $NO_CHUNKED_DIR
# python ./scripts/feature_extraction/edaic/split_into_chunks.py --source-dir $NO_CHUNKED_DIR --modality-id video_pose_gaze_aus --no-idxs-id no_face_idxs --dest-dir /home/sn666/processed_data/E-DAIC/data/ --nseconds 60 --frame-rate 25
