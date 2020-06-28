# clear existing mlt_submission folder
rm -r mlt_submission
mkdir -pv mlt_submission/{src,model}

# copy requirements.txt
cp -v requirements.txt mlt_submission/
ls -la mlt_submission/requirements.txt


# copy model file
mkdir -pv mlt_submission/model
cp -v /ext/signate_edge_ai/model/resnet101_csv_15.5classes.all_bboxes.h5.frozen mlt_submission/model/

# copy source files
cp -v src/main.py                           mlt_submission/src/
cp -v src/predictor.py                      mlt_submission/src/
cp -v src/object_tracker.py                 mlt_submission/src/

if [ -f mlt_submission.zip ]; then
    rm mlt_submission.zip
    zip -q -r mlt_submission_`date +"%m%d%H%m"`.zip mlt_submission
fi