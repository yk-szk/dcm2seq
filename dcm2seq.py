import argparse
from pathlib import Path
from datetime import datetime
import tempfile
import shutil
import numpy as np
import pandas as pd
import tqdm
import pydicom
from pydicom.valuerep import DT
import SimpleITK as sitk

from logging import basicConfig, getLogger, Formatter, FileHandler, INFO

basicConfig(level=INFO, format='%(asctime)s %(levelname)s :%(message)s')
log_format = Formatter('%(asctime)s %(levelname)s :%(message)s')
logger = getLogger(__name__)


def verbosity_to_level(verbosity):
    return max(0, 30 - 10 * verbosity)


def main():
    parser = argparse.ArgumentParser(description='argparse example.')
    parser.add_argument('input', help="Root directory", metavar='<input>')
    parser.add_argument('output', help="Output directory", metavar='<output>')
    parser.add_argument('--ext',
                        help='File extension. default: %(default)s',
                        metavar='str',
                        default='.mha')
    parser.add_argument(
        '--prefix',
        help='Prefix of the output filename. default: %(default)s',
        metavar='str',
        default='SE')
    parser.add_argument('--compress',
                        help='Compress the output image. default: %(default)s',
                        type=str,
                        choices=['auto', 'true', 'false'],
                        default='auto')
    parser.add_argument('--offset',
                        help='Offset to the number. default: %(default)s',
                        type=int,
                        metavar='int',
                        default=1)
    parser.add_argument('--logdir',
                        help='Directory to store logs. default: %(default)s',
                        metavar='str',
                        default=None)
    parser.add_argument('--verbose',
                        help='Verbosity. default: %(default)s',
                        type=int,
                        metavar='level',
                        default=0)

    args = parser.parse_args()

    logger.setLevel(verbosity_to_level(args.verbose))
    if args.logdir is not None:
        logdir = Path(args.logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        handler = FileHandler(
            logdir /
            '{}.log'.format(datetime.today().strftime("%y%m%d_%H%M%S")))
        handler.setLevel(verbosity_to_level(args.verbose))
        handler.setFormatter(log_format)
        logger.addHandler(handler)

    if args.input.endswith('.zip'):
        tempdir = tempfile.TemporaryDirectory()
        logger.info(tempdir.name)
        shutil.unpack_archive(args.input, tempdir.name)
        root_dir = Path(tempdir.name)
    else:
        root_dir = Path(args.input)
    out_dir = Path(args.output)

    compression = {'auto': None, 'true': True, 'false': False}[args.compress]
    dtype = None
    prefix = args.prefix
    ext = args.ext
    offset = args.offset

    logger.info('Collect dicom information')
    all_files = [
        str(e) for e in tqdm.tqdm(root_dir.glob('**/*'), desc='list all files')
        if e.is_file()
    ]

    key_tags = [
        'PatientID', 'SeriesInstanceUID', 'AcquisitionDate', 'AcquisitionTime',
        'ImageOrientationPatient', 'ImagePositionPatient'
    ]
    dcm_files = []
    for fn in tqdm.tqdm(all_files):
        try:
            dcm = pydicom.dcmread(fn, stop_before_pixels=True)
            for tag in key_tags:
                if not hasattr(dcm, tag):
                    raise RuntimeError(tag + 'Not found.')
            dcm_files.append([fn] + [dcm.get(tag) for tag in key_tags])
        except Exception as e:
            logger.warning({'filename': fn, 'exception': e})

    df = pd.DataFrame(dcm_files, columns=['filepath'] + key_tags)

    logger.info('Convert dicom files')

    def sort_dicom(df):
        try:
            orientation = np.array(
                df['ImageOrientationPatient'].iloc[0]).reshape((2, 3))
        except Exception as e:
            print(np.array(df['ImageOrientationPatient'].iloc[0]))
            return df
        third_axis = np.cross(orientation[0], orientation[1])
        locs = df['ImagePositionPatient'].map(lambda p: np.dot(third_axis, p))
        sorted_index = np.argsort(locs)
        return df.iloc[sorted_index]

    FLOAT_TYPES = set([
        sitk.sitkFloat32, sitk.sitkFloat64, sitk.sitkVectorFloat32,
        sitk.sitkVectorFloat64
    ])

    for patient_id, df_patient in df.groupby('PatientID'):
        logger.info(patient_id)
        sids, times = [], []
        for series_id, df_series in df_patient.groupby('SeriesInstanceUID'):
            sids.append(series_id)
            dts = df_series.apply(
                lambda row: DT(row.AcquisitionDate + row.AcquisitionTime),
                axis=1).tolist()
            if len(df_series) <= 2:
                times.append(dts[0])
            else:
                dts.sort()
                times.append(dts[len(dts) // 2])
        nums = np.argsort(np.argsort(times))
        series_id2series_number = dict(zip(sids, nums))

        for series_id, df_series in df_patient.groupby('SeriesInstanceUID'):
            logger.debug(series_id)
            output_filename = out_dir / patient_id / (prefix + '{:d}'.format(
                series_id2series_number[series_id] + offset) + ext)
            output_filename.parent.mkdir(parents=True, exist_ok=True)
            filenames = sort_dicom(df_series)['filepath'].tolist()
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(filenames)
            image = reader.Execute()
            if image.GetPixelID() == sitk.sitkFloat64 and dtype is None:
                f = sitk.CastImageFilter()
                f.SetOutputPixelType(sitk.sitkFloat32)
                image = f.Execute(image)
            writer = sitk.ImageFileWriter()
            if compression is None:
                compression = image.GetPixelID() not in FLOAT_TYPES
            writer.SetUseCompression(compression)
            writer.SetFileName(str(output_filename))
            writer.Execute(image)

    logger.info('End')


import sys
if __name__ == "__main__":
    sys.exit(main())