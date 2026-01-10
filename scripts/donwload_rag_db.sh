hf download blueyo0/monai_rag_db --local-dir crea/monai_rag/monai_rag_zip --repo-type dataset
unzip -o crea/monai_rag/monai_rag_zip/monai_rag_db.zip -d crea/monai_rag/
rm -rf crea/monai_rag/monai_rag_zip
python -m crea.monai_rag.query_rag_db