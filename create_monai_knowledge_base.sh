pushd meddle/monai_rag
rm -rf docs.monai.io
bash wget_monai_doc.sh
python create_monai_rag_db.py
popd