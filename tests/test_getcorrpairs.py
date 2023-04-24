from module1_s4_functions import *
import pandas as pd #type:ignore
import numpy as np
test_array=pd.DataFrame(
    np.array([[1,2,1,1],[2,4,0,2],[3,6,-1,-1],[4,8,-2,2]]),
    columns=['a','b','c','d']
    )
def test_pos_corr():
    '''Tests get_correlation_pairs against a test matrix for positive corr'''
    result_array=pd.DataFrame(
    {'r-value':1.0, 'feature_pair':[frozenset(('a','b'))]}
    )
    positive_result=get_correlation_pairs(test_array,positive_cut_off=0.7)
    pd.testing.assert_frame_equal(positive_result,result_array)
def test_neg_corr():
    '''Tests get_correlation_pairs against a test matrix for negative corr'''
    result_array=pd.DataFrame(
    {'r-value':[-1.0,-1.0], 'feature_pair':[frozenset(('c','a')),frozenset(('c','b'))]}
    )
    negative_result=get_correlation_pairs(test_array,negative_cut_off=-0.7)
    pd.testing.assert_frame_equal(negative_result,result_array)
def test_non_corr():
    '''Tests get_correlation_pairs against a test matrix for non-corr'''
    result_array=pd.DataFrame(
    {'r-value':[0.0,0.0,0.0], 'feature_pair':[frozenset(('a','d')),frozenset(('b','d')),frozenset(('c','d'))]}
    )
    non_result=get_correlation_pairs(test_array,positive_cut_off=0.3,negative_cut_off=-0.3,leave_center=True)
    pd.testing.assert_frame_equal(non_result,result_array)   
