#!/bin/bash
# load Python environment
# module load jump

/home/zyuan1/test_jumplib/JUMPspecLib-main/jump_lib -pp_s jumplib_search.params
/home/zyuan1/test_jumplib/JUMPspecLib-main/jump_lib -f jumplib_filter.params
/home/zyuan1/test_jumplib/JUMPspecLib-main/jump_lib -q jump_lib_q.params
