import SimpleITK as sitk

reader = sitk.ImageSeriesReader()


def read(path):
    dicoms = reader.GetGDCMSeriesFileNames(str(path))
    reader.SetFileNames(dicoms)
    image = reader.Execute()
    # image_array = sitk.GetArrayFromImage(image)
    # image_size = image.GetSize()
    return image


def read_tag(path, key):
    image = sitk.ReadImage(str(path))
    # keys = image.GetMetaDataKeys()
    # for key in keys:
    #     print(key + " " + image.GetMetaData(key))
    return image.GetMetaData(key)


def write(image, path):
    sitk.WriteImage(image, str(path))


if __name__ == "__main__":
    # image = read(
    #     'E:/SourceDatasets/Neurofibromatosis/NF/WBMRI_008_00001_20070516/DICOM/Series36-000094--T1-COMPOSED')
    i = read_tag(
        'E:/SourceDatasets/Neurofibromatosis/NF/2645145_10292687_20070314/DICOM/Series22-000008--t1-composite/001-0007.dcm',
        '0018|0024')
    # stir 0018|0024 *tir2d1rr25
    # t1 0018|0024 *tse2d1_3
    # pixel spacing 0028|0030 1.5625\1.5625
    # thickness 0018|0050 10

    print(i)
