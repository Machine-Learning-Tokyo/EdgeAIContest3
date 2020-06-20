# clear existing sample_submit folder
rm -r sample_submit
mkdir -pv sample_submit/src

# copy requirements.txt
cp -v requirements.txt sample_submit/
ls -la sample_submit/requirements.txt

cp -v src/predictor.py                      sample_submit/src/
cp -v src/object_tracker.py                 sample_submit/src/
cp -v src/main.py                           sample_submit/src/

# copy model file
mkdir -pv sample_submit/model
cp -v /ext/signate_edge_ai/model/resnet50_csv_01.h5.frozen sample_submit/model/

zip -q -r sample_submit.zip sample_submit