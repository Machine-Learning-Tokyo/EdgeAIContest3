# clear existing sample_submit folder
rm -r sample_submit
mkdir -pv sample_submit/src

# copy requirements.txt
cp -v requirements.txt sample_submit/
ls -la sample_submit/requirements.txt
#git clone https://github.com/fizyr/keras-retinanet.git
#cd keras-retinanet/
#python setup.py build_ext --inplace
#cd ..

cp -r src/keras_retinanet       sample_submit/src/
cp -v src/predictor.py                      sample_submit/src/
cp -v src/object_tracker.py                 sample_submit/src/
cp -v src/process_video.py                  sample_submit/src/
cp -v src/retinanet_wrapper.py              sample_submit/src/
cp -v src/signate_sub.py                    sample_submit/src/
cp -v src/stabilizer.py                     sample_submit/src/
cp -v src/main.py                           sample_submit/src/

# copy model file
mkdir -pv sample_submit/model
cp -v model/resnet50_csv_01.h5.frozen sample_submit/model/

zip -q -r sample_submit.zip sample_submit