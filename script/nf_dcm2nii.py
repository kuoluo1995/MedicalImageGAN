from pathlib import Path
from utils import dcm_utils, yaml_utils

output_path = 'E:/Dataset/Neurofibromatosis/source2018'

A = 'STIR'
B = 'T1'
sequence_map = {'*tir2d1rr13': A, '*se2d1': B, '*tir2d1rr25': A, '*tse2d1_3': B, 'tir2d1rr25': A, 'tse2d1_3': B}
tag_map = {'0008|0012': 'instance_creation_date', '0018|0024': 'sequence_name', '0028|0030': 'pix_spacing',
           '0018|0050': 'thickness', '0010|0020': 'patient_id'}

error = dict()
error['read dcm error'] = list()
error['match datas error'] = list()
error['sequence name error'] = list()


def dcm2nii(source_data):
    flag = False
    dict_ = dict()
    for type_ in Path(source_data).iterdir():
        tag_dict = dict()
        for slice in type_.iterdir():
            if slice.suffix == '.dcm':
                for key, value in tag_map.items():
                    try:
                        tag_dict[value] = dcm_utils.read_tag(str(slice), key).strip()
                    except:
                        error['read dcm error'].append(str(slice))
                        flag = True
                        break
                break
        image = dcm_utils.read(str(type_))
        shape = image.GetSize()
        tag_dict['shape'] = (shape[0], shape[1], shape[2])
        tag_dict['path'] = str(type_)
        if not flag:
            try:
                dict_[sequence_map[tag_dict['sequence_name']]] = tag_dict
            except:
                error['sequence name error'].append(tag_dict['path'])

    if A in dict_.keys() and B in dict_.keys():
        for key, value in dict_.items():
            print(output_path + '/' + key + '/' + value['patient_id'] + '_' + value['instance_creation_date'] + '.nii')
            Path(output_path + '/' + key + '/').mkdir(parents=True, exist_ok=True)
            dcm_utils.write(dcm_utils.read(value['path']),
                            output_path + '/' + key + '/' + value['patient_id'] + '_' + value[
                                'instance_creation_date'] + '.nii')
        return dict_
    else:
        one_error = list()
        for key, value in dict_.items():
            one_error.append(key + ':' + value['path'])
        error['match datas error'].append(one_error)
        return None


if __name__ == '__main__':
    source_data = 'E:/SourceDatasets/Neurofibromatosis/NF-2018'
    detail = list()
    for dataset in Path(source_data).iterdir():
        dataset = dataset / 'DICOM'
        result = dcm2nii(dataset)
        if result is not None:
            detail.append(result)
    yaml_utils.write(output_path + '/info.yaml', detail)
    yaml_utils.write(output_path + '/error.yaml', error)
