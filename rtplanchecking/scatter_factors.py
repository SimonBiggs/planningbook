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

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

from .pull_mephisto import create_pdd_function


def create_scatter_functions(scatter_filepath=None,
                             pdd_filepath=None,
                             SSD_data=100,
                             SSD_cal=100,
                             calibration_field_area=100):
    scatter_factor_data = pd.DataFrame.from_csv(scatter_filepath)
    open_pdd = create_pdd_function(pdd_filepath, SSD=SSD_data)

    field_area = np.array(scatter_factor_data.index)

    Scp_open_data = np.array(scatter_factor_data['Scp_open'])
    Scp_open_ref = ~np.isnan(Scp_open_data)

    Sc_open_data = np.array(scatter_factor_data['Sc_open'])
    Sc_open_ref = ~np.isnan(Sc_open_data)

    def Scp(iso_area):
        interpolation = interp1d(
            np.sqrt(field_area[Scp_open_ref]), Scp_open_data[Scp_open_ref])

        return interpolation(np.sqrt(iso_area))

    def Sc(iso_area):
        interpolation = interp1d(
            np.sqrt(field_area[Sc_open_ref]), Sc_open_data[Sc_open_ref])

        return interpolation(np.sqrt(iso_area))

    def Sp_not_normalised(area_surface):
        area_reference = calibration_field_area * (SSD_data / SSD_cal) ** 2

        area_iso = area_surface / (SSD_data / SSD_cal) ** 2
        Sp_uncorr = Scp(area_iso) / Sc(area_iso)
        pdd_correction = (
            open_pdd(10, area_reference) / open_pdd(10, area_surface))
        return Sp_uncorr * np.squeeze(pdd_correction)

    def Sp(area_surface):
        return (
            Sp_not_normalised(area_surface) /
            Sp_not_normalised(calibration_field_area))

    return {'Scp': Scp, 'Sc': Sc, 'Sp': Sp}


def create_wedge_functions(scatter_filepath=None,
                           pdd_filepath=None,
                           SSD_data=100,
                           SSD_cal=100,
                           calibration_field_area=100):
    scatter_factor_data = pd.DataFrame.from_csv(scatter_filepath)
    wedge_pdd = create_pdd_function(pdd_filepath, SSD=SSD_data)

    field_area = np.array(scatter_factor_data.index)

    Scp_wedge_data = np.array(scatter_factor_data['Scp_wedge'])
    Scp_wedge_ref = ~np.isnan(Scp_wedge_data)

    Sc_wedge_data = np.array(scatter_factor_data['Sc_wedge'])
    Sc_wedge_ref = ~np.isnan(Sc_wedge_data)

    def Scp_wedge(iso_area):
        interpolation = interp1d(
            np.sqrt(field_area[Scp_wedge_ref]), Scp_wedge_data[Scp_wedge_ref])

        return interpolation(np.sqrt(iso_area))

    def Sc_wedge(iso_area):
        interpolation = interp1d(
            np.sqrt(field_area[Sc_wedge_ref]), Sc_wedge_data[Sc_wedge_ref])

        return interpolation(np.sqrt(iso_area))

    def Sp_not_normalised(area_surface):
        area_reference = calibration_field_area * (SSD_data / SSD_cal) ** 2

        area_iso = area_surface / (SSD_data / SSD_cal) ** 2
        Sp_uncorr = Scp_wedge(area_iso) / Sc_wedge(area_iso)
        pdd_correction = (
            wedge_pdd(10, area_reference) / wedge_pdd(10, area_surface))
        return Sp_uncorr * np.squeeze(pdd_correction)

    def Sp_wedge(area_surface):
        return (
            Sp_not_normalised(area_surface) /
            Sp_not_normalised(calibration_field_area))

    return {'Scp_wedge': Scp_wedge, 'Sc_wedge': Sc_wedge, 'Sp_wedge': Sp_wedge}
