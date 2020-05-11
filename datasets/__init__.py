from .loadDataSet import (load_gunpoint)

from .ucr import ucr_dataset_info, ucr_dataset_list, fetch_ucr_dataset

__all__ = ['load_basic_motions', 'load_coffee', 'load_gunpoint',
           'load_pig_central_venous_pressure', 'make_cylinder_bell_funnel',
           'ucr_dataset_info', 'ucr_dataset_list', 'fetch_ucr_dataset',
           'uea_dataset_info', 'uea_dataset_list', 'fetch_uea_dataset']