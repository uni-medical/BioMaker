# 使用wget镜像整个文档（保持目录结构）
wget --mirror --convert-links --adjust-extension --page-requisites --no-parent --wait=1 -e robots=off -c https://docs.monai.io/en/1.4.0/api.html

# remove useless files
rm docs.monai.io/en/latest/_*/* -rf
