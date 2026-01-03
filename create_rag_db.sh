pushd crea/monai_rag
rm -rf docs.monai.io
bash wget_monai_doc.sh # MONAI-1.4.0
python create_monai_rag_db.py
popd