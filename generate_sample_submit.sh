# clear existing sample_submit folder
rm -r sample_submit
mkdir -pv sample_submit/{src,model}

# copy requirements.txt
cp -v requirements.txt sample_submit/
ls -la sample_submit/requirements.txt


# copy model file
mkdir -pv sample_submit/model
cp -v /ext/signate_edge_ai/model/resnet50_csv_01.h5.frozen sample_submit/model/

# copy source files
cp -v src/main.py                           sample_submit/src/
cp -v src/predictor.py                      sample_submit/src/
cp -v src/object_tracker.py                 sample_submit/src/
