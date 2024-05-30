#!/bin/bash
source ~/.bashrc

set -e
cd ../models/pointcept

echo "pointcept: [1/4] installing dependencies..."

apt-get -qq update
apt-get -qq install libgl1-mesa-glx

echo "pointcept: [2/4] initializing environment..."

conda env create -f environment.yml

echo "pointcept: [3/4] initializing environment..."

conda activate pointcept

cd libs/pointops
python setup.py install

conda deactivate

echo "pointcept: [4/4] downloading pre-trained weights..."

cd ../..

wget -q 'https://southeastasia1-mediap.svc.ms/transform/zip?cs=fFNQTw' -O model.zip \
    --post-data='zipFileName=s3dis-semseg-pt-v3m1-0-rpe.zip&guid=71104f44-ff52-4030-86f0-1ae7d3afdfb4&provider=spo&files=%7B%22items%22%3A%5B%7B%22name%22%3A%22s3dis-semseg-pt-v3m1-0-rpe%22%2C%22size%22%3A0%2C%22docId%22%3A%22https%3A%2F%2Funivindonesia-my.sharepoint.com%3A443%2F_api%2Fv2.0%2Fdrives%2Fb%21-_UjHOI7oEmXw0CXf3CHcdDk0BvDJMlBpKEyTi_1oOLXjv7dI4hVQrjPbpmPfQG4%2Fitems%2F01QJVOALCZRJLXV3VHKRH3O4I3TJNLXQOZ%3Fversion%3DPublished%26access_token%3DeyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOiIwMDAwMDAwMy0wMDAwLTBmZjEtY2UwMC0wMDAwMDAwMDAwMDAvdW5pdmluZG9uZXNpYS1teS5zaGFyZXBvaW50LmNvbUA0ODVkMGMyYS1iM2JjLTQwN2MtOThmYi04MjU0MDgyNTg2NTYiLCJjYWNoZWtleSI6IjBoLmZ8bWVtYmVyc2hpcHwxMDAzMjAwMGRmZTFkMzMxQGxpdmUuY29tIiwiZW5kcG9pbnR1cmwiOiJUZ2VwKzBXT3prbkFSUzZ6aytIOE1uWGFjYVRiQm96d2FGU2VYVG9PNkdFPSIsImVuZHBvaW50dXJsTGVuZ3RoIjoiMTIzIiwiZXhwIjoiMTcxNzA0ODgwMCIsImlwYWRkciI6IjEwMy4xMjEuMTgyLjE4MiIsImlzbG9vcGJhY2siOiJUcnVlIiwiaXNzIjoiMDAwMDAwMDMtMDAwMC0wZmYxLWNlMDAtMDAwMDAwMDAwMDAwIiwiaXN1c2VyIjoidHJ1ZSIsIm5hbWVpZCI6IjAjLmZ8bWVtYmVyc2hpcHxtYXJjZWxsaW5vLmNocmlzQG9mZmljZS51aS5hYy5pZCIsIm5iZiI6IjE3MTcwMjcyMDAiLCJuaWkiOiJtaWNyb3NvZnQuc2hhcmVwb2ludCIsInNpZCI6ImI5YzA5N2U0LTY4MjMtNDRjMi1iZmZiLTc2N2U1NGFlYjhiMSIsInNpZ25pbl9zdGF0ZSI6IltcImttc2lcIl0iLCJzaXRlaWQiOiJNV015TTJZMVptSXRNMkpsTWkwME9XRXdMVGszWXpNdE5EQTVOemRtTnpBNE56Y3giLCJzbmlkIjoiNiIsInN0cCI6InQiLCJ0dCI6IjAiLCJ2ZXIiOiJoYXNoZWRwcm9vZnRva2VuIn0.DYnFUEClBwt1qfxSz4Mfhr6WvyYTf8a5X2eUTh0RRRw%22%2C%22isFolder%22%3Atrue%7D%5D%7D&oAuthToken=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6IkwxS2ZLRklfam5YYndXYzIyeFp4dzFzVUhIMCIsImtpZCI6IkwxS2ZLRklfam5YYndXYzIyeFp4dzFzVUhIMCJ9.eyJhdWQiOiJodHRwczovL3NvdXRoZWFzdGFzaWExLW1lZGlhcC5zdmMubXMiLCJpc3MiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC80ODVkMGMyYS1iM2JjLTQwN2MtOThmYi04MjU0MDgyNTg2NTYvIiwiaWF0IjoxNzE3MDM1NDA0LCJuYmYiOjE3MTcwMzU0MDQsImV4cCI6MTcxNzAzOTY5MywiYWNyIjoiMSIsImFpbyI6IkUyTmdZR0NjY0s5Zi9DWC91OXlVbSt0U2srMk45di94WGpDcDB1Q0FZMFhJQVp2bS9jSUEiLCJhbXIiOlsicHdkIl0sImFwcF9kaXNwbGF5bmFtZSI6Ik9mZmljZSAzNjUgU2hhcmVQb2ludCBPbmxpbmUiLCJhcHBpZCI6IjAwMDAwMDAzLTAwMDAtMGZmMS1jZTAwLTAwMDAwMDAwMDAwMCIsImFwcGlkYWNyIjoiMiIsImF1dGhfdGltZSI6MTcxNjkwOTY0MSwiaXBhZGRyIjoiMTAzLjEyMS4xODIuMTgyIiwibmFtZSI6Ik1hcmNlbGxpbm8gQ2hyaXMgT1x1MDAyN1ZhcmEiLCJvaWQiOiI2ZDA0YjMyOS1iMDg4LTRkZjYtYTNkYi01MDJjMGU3NTAyNmMiLCJwdWlkIjoiMTAwMzIwMDBERkUxRDMzMSIsInJoIjoiMC5BU29BS2d4ZFNMeXpmRUNZLTRKVUNDV0dWdEVMVDVSN0VSeExyeWFBVHRsZWRuNHFBUHcuIiwic2NwIjoiU2l0ZXMubWFuYWdlLkFsbCIsInN1YiI6IjJwVE5JOFhRVFhyQl9BSWVCc2N5emcwRU1ScjczNW13VkI1dUlkSkdPZWsiLCJ0aWQiOiI0ODVkMGMyYS1iM2JjLTQwN2MtOThmYi04MjU0MDgyNTg2NTYiLCJ1bmlxdWVfbmFtZSI6Im1hcmNlbGxpbm8uY2hyaXNAb2ZmaWNlLnVpLmFjLmlkIiwidXBuIjoibWFyY2VsbGluby5jaHJpc0BvZmZpY2UudWkuYWMuaWQiLCJ1dGkiOiJxbmN0LXRXZmNVQ2FXMVBlTVBIZUFBIiwidmVyIjoiMS4wIn0.WAK1SH27ZKoheajOQyJG6NgRBHHOmaqFctztLDkyH1kQ5cQnaG1jfx86Jlt_AIq1wQusa3RstPeFci-dIuu9WKCqJg6xaf2YKtdtul_b3XqsqWsPD4TaLzTb74x7oV91NX3dSkvkgeFdzjO7EpW81KyU_z6WuYc668Kt2iKxD8oT8p6hDP3z10JjHsUjrVAXZk6BCbnoKBppL7jP1LjBgzpm6T-4-X88z5-13uqMCtGr-dPDFIEKGBO2TT_SkX5ZU2uQot_agsTUbP5QFLncXKCCM513YSGCF_XJJ4J_k-5ajmmPLISKMOMLvLH2Zn5p-7PL62P36hPey8SdDtZBrA'

unzip model.zip
rm model.zip

mkdir models
mv s3dis-semseg-pt-v3m1-0-rpe models/ptv3
