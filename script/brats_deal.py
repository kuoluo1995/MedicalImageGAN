from pathlib import Path
from utils import nii_utils, yaml_utils

output_path = Path('E:/Datasets/BraTS_2018')


def nii_gz2nii(path):
    detail = dict()
    for file in Path(path).iterdir():
        item_name = Path(path).stem
        class_fold = file.stem.replace(item_name, '').replace('.nii', '').replace('_', '')
        output = output_path / class_fold / (item_name + '.nii')
        output.parent.mkdir(parents=True, exist_ok=True)
        image = nii_utils.nii_reader(str(file))
        header = nii_utils.nii_header_reader(str(file))
        nii_utils.nii_writer(str(output), header, image)
        detail[class_fold] = image.shape
    return detail


if __name__ == '__main__':
    source_data = 'E:/SourceDatasets/LGG'
    _dict = dict()
    for item in Path(source_data).iterdir():
        _dict[item.stem] = nii_gz2nii(str(item))
        print('\r>>'+item.stem, end='')
    yaml_utils.write(str(output_path) + '/detail.yaml', _dict)
