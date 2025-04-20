# 使用wget镜像整个文档（保持目录结构）
wget --mirror --convert-links --adjust-extension --page-requisites --no-parent --wait=1 -e robots=off https://docs.monai.io/en/latest/api.html

# remove useless files
rm docs.monai.io/en/latest/_*/* -rf
