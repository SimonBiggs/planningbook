# Copyright (C) 2015 Simon Biggs
# This program is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public
# License as published by the Free Software Foundation, either
# version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public
# License along with this program. If not, see
# http://www.gnu.org/licenses/.

import re
import numpy as np
from scipy.interpolate import RectBivariateSpline


def create_pdd_function(filepath,
                        mephisto_field_size_defined_at=100,
                        SSD=100):

    with open(filepath) as file:
        lines = np.array(file.readlines())

    begin_scan_index = np.array([
            i for i, item in enumerate(lines)
            if re.search('.*\tBEGIN_SCAN  .*', item)]).astype(int)
    end_scan_index = np.array([
            i for i, item in enumerate(lines)
            if re.search('.*\tEND_SCAN  .*', item)]).astype(int)

    field_inplane_index = np.array([
            i for i, item in enumerate(lines)
            if re.search('.*\t\tFIELD_INPLANE.*', item)]).astype(int)
    field_crossplane_index = np.array([
            i for i, item in enumerate(lines)
            if re.search('.*\t\FIELD_CROSSPLANE.*', item)]).astype(int)

    field_inplane = np.array([
            re.search('.*\t\tFIELD_INPLANE=(\d+\.\d+).*', item).group(1)
            for item in lines[field_inplane_index]
        ]).astype(float) / 10 * (SSD / mephisto_field_size_defined_at)

    field_crossplane = np.array([
            re.search('.*\t\FIELD_CROSSPLANE=(\d+\.\d+).*', item).group(1)
            for item in lines[field_crossplane_index]
        ]).astype(float) / 10 * (SSD / mephisto_field_size_defined_at)

    field_area = field_inplane * field_crossplane

    begin_data_index = np.array([
            i for i, item in enumerate(lines)
            if re.search('.*BEGIN_DATA.*', item)]).astype(int)
    end_data_index = np.array([
            i for i, item in enumerate(lines)
            if re.search('.*END_DATA.*', item)]).astype(int)

    data_index = [
        range(begin_data_index[i]+1, end_data_index[i])
        for i in range(len(begin_data_index))]

    depth = np.array([
        re.search('\t\t\t(\d+\.\d+)\t\t(\d+\.\d+E[-+]?\d+)\n', item).group(1)
        for item in lines[data_index[0]]
    ]).astype(float)

    depth_cm = depth / 10

    for index in data_index:
        test_depth = np.array([
            re.search(
                '\t\t\t(\d+\.\d+)\t\t(\d+\.\d+E[-+]?\d+)\n', item).group(1)
            for item in lines[index]
        ]).astype(float)

        assert np.all(test_depth == depth)

    assert np.all(
        (begin_data_index > begin_scan_index) &
        (begin_data_index < end_scan_index) &
        (end_data_index > begin_scan_index) &
        (end_data_index < end_scan_index)
    )

    data_length = len(data_index[0])
    for index in data_index:
        assert len(index) == data_length

    pdd_data = np.zeros([data_length, len(field_area)])

    for i, index in enumerate(data_index):
        reading = np.array([
            re.search(
                '\t\t\t(\d+\.\d+)\t\t(\d+\.\d+E[-+]?\d+)\n', item).group(2)
            for item in lines[index]
        ]).astype(float)

        pdd_data[:, i] = reading / np.max(reading) * 100

    spline = RectBivariateSpline(
        depth_cm, field_area, pdd_data, kx=1, ky=1, s=0)

    return spline
