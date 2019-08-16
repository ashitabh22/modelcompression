import numpy as np
from scipy import stats
from statistics import median,mean
import random
import sys
import math
from scipy.linalg import circulant
from scipy.linalg import toeplitz
from scipy.linalg import hankel
from scipy.stats.mstats import winsorize
from timeit import default_timer as timer
from operator import add
from functools import reduce


def create_matrix(rows,cols):

    a=[[random.randint(1,10) for i in range(cols)] for j in range(1,rows*cols, cols) ]
    # a=[[10,3,10,1,3,1,2],
    #    [8,10,6,8,9,1,3],
    #    [5,2,3,9,10,1,4],
    #    [10,9,4,3,6,1,5],
    #    [3,7,2,1,4,1,6]]

    print(a)
    return a
def tobc(curr_arr,block_size,central_tendency,cutting_ratio,percentile):
    answer =[]
    for curr_col in range(np.shape(curr_arr)[1]):
        curr_row =0
        curr_col_moving = curr_col
        sum_=0
        median_list = []
        while curr_row < np.shape(curr_arr)[0]:
            sum_ = sum_+ curr_arr[curr_row][curr_col_moving]
            median_list.append(curr_arr[curr_row][curr_col_moving])
            curr_col_moving = curr_col_moving+1
            if curr_col_moving ==np.shape(curr_arr)[1]:
                curr_col_moving =0
            curr_row= curr_row+1
        if central_tendency == "sum":
            temp_var= sum_
        if central_tendency == "mean":
            temp_var = sum_/block_size[0]
        if central_tendency == "median":
            temp_var = median(median_list)
        if central_tendency == "trim_mean":
            temp_var = stats.trim_mean(np.array(median_list),cutting_ratio)
        if central_tendency == "percentile":
            temp_var = np.percentile(np.array(median_list),percentile)
        if central_tendency=="win_mean":
            temp_var = mean(winsorize(np.array(median_list),cutting_ratio))
        answer.append(temp_var)
    mid_answer = np.copy(circulant(answer).transpose())
    return mid_answer

def to_hankel(curr_arr,central_tendency,cutting_ratio,percentile):
    upper=[]
    lower=[]
    df=[]
    curr_arr = np.array(curr_arr)
    curr_arr = curr_arr.astype(float)
    # diags = [matrix[::-1,:].diagonal(i) for i in range(-3,4)]
    diags = [curr_arr[::-1,:].diagonal(i) for i in range(-curr_arr.shape[0]+1,curr_arr.shape[1])]

    dlist=[n.tolist() for n in diags]
    for elements in dlist:
        if central_tendency == "mean":
            temp=sum(elements)/len(elements)
        if central_tendency == "median":
            temp= median(elements)
        if central_tendency == "trim_mean":
            temp= stats.trim_mean(np.array(elements),cutting_ratio)
        if central_tendency == "percentile":
            temp= np.percentile(np.array(elements),percentile)
        if central_tendency=="win_mean":
            temp= mean(winsorize(np.array(elements),cutting_ratio))
        df.append(temp)
    # print(df)

    # print(dlist)
    # print(len(df))
    for i in range(len(df)):
        if(i<(math.floor(len(df)/2))):
            upper.append(df[i])
        elif(i==math.floor(len(df)/2)):
            upper.append(df[i])
            lower.append(df[i])
        else:
            lower.append(df[i])
    # print(upper)
    # print(lower)
    # new_arr=hankel(upper,lower).transpose()
    # return new_arr
    mid_answer = np.copy(hankel(upper,lower).transpose())
    return mid_answer
# def toeplitz_faster(curr_arr,central_tendency,cutting_ratio,percentile):
#   upper=[]
#   lower=[]
#   df=[]
#   curr_arr = np.array(curr_arr)
#   curr_arr = curr_arr.astype(float)
#   diags=[curr_arr.diagonal(i) for i in range(curr_arr.shape[1]-1,-curr_arr.shape[0],-1)]
#   dlist=[n.tolist() for n in diags]
#   for elements in dlist:
#       if central_tendency == "mean":
#           temp=sum(elements)/len(elements)
#       if central_tendency == "median":
#           temp= median(elements)
#       if central_tendency == "trim_mean":
#           temp= stats.trim_mean(np.array(elements),cutting_ratio)
#       if central_tendency == "percentile":
#           temp= np.percentile(np.array(elements),percentile)
#       if central_tendency=="win_mean":
#           temp= mean(winsorize(np.array(elements),cutting_ratio))
#       df.append(temp)
#   for i in range(len(df)):
#       if(i<(math.floor(len(df)/2))):
#           upper.append(df[i])
#       elif(i==math.floor(len(df)/2)):
#           upper.append(df[i])
#           lower.append(df[i])
#       else:
#           lower.append(df[i])
#   # print(upper)
#   # print(lower)
#   # new_arr=hankel(upper,lower).transpose()
#   # return new_arr
#   upper.reverse()
#   mid_answer = np.copy(toeplitz(upper,lower).transpose())
#   return mid_answer

def toeplitz_faster(curr_arr,central_tendency,cutting_ratio,percentile):
    upper=[]
    lower=[]
    df=[]
    curr_arr = np.array(curr_arr)
    rows,col=np.shape(curr_arr)
    curr_arr = curr_arr.astype(float)
    diags=[curr_arr.diagonal(i) for i in range(curr_arr.shape[1]-1,-curr_arr.shape[0],-1)]
    dlist=[n.tolist() for n in diags]
    for elements in dlist:
        if central_tendency == "mean":
            temp=sum(elements)/len(elements)
        if central_tendency == "median":
            temp= median(elements)
        if central_tendency == "trim_mean":
            temp= stats.trim_mean(np.array(elements),cutting_ratio)
        if central_tendency == "percentile":
            temp= np.percentile(np.array(elements),percentile)
        if central_tendency=="win_mean":
            temp= mean(winsorize(np.array(elements),cutting_ratio))
        df.append(temp)
    for i in range(len(df)):
        if(i<len(df)-rows):
            upper.append(df[i])
        elif(i==len(df)-rows):
            upper.append(df[i])
            lower.append(df[i])
        else:
            lower.append(df[i])
    # print(upper)
    # print(lower)
    # new_arr=hankel(upper,lower).transpose()
    # return new_arr
    upper.reverse()
    mid_answer = np.copy(toeplitz(upper,lower).transpose())
    return mid_answer

def to_toeplitz(curr_arr,central_tendency,cutting_ratio,percentile):
    # curr_arr = np.array(curr_arr)
    # curr_arr = curr_arr.astype(float)
    #print(curr_arr)
    upper=[]
    lower=[]
    rows,cols = np.shape(curr_arr)
    i=0
    j=0
    dsum=0
    dlist=[]
    temp=0
    for i in range(np.shape(curr_arr)[0]):
        for j in range(np.shape(curr_arr)[1]):
            if(i==j):
                dlist.append(curr_arr[i][j])
                if central_tendency == "mean":
                    temp=sum(dlist)/len(dlist)
                if central_tendency == "median":
                    temp= median(dlist)
                if central_tendency == "trim_mean":
                    temp= stats.trim_mean(np.array(dlist),cutting_ratio)
                if central_tendency == "percentile":
                    temp= np.percentile(np.array(dlist),percentile)
                if central_tendency=="win_mean":
                    temp= mean(winsorize(np.array(dlist),cutting_ratio))
    upper.append(temp)
    lower.append(temp)
    temp=0
    dlist.clear()
    for k in range(1,np.shape(curr_arr)[1]):
        for i in range(np.shape(curr_arr)[0]):
            for j in range(np.shape(curr_arr)[1]):
                if(j==i+k):
                    dlist.append(curr_arr[i][j])
                    if central_tendency == "mean":
                        temp=sum(dlist)/len(dlist)
                    if central_tendency == "median":
                        temp= median(dlist)
                    if central_tendency == "trim_mean":
                        temp= stats.trim_mean(np.array(dlist),cutting_ratio)
                    if central_tendency == "percentile":
                        temp= np.percentile(np.array(dlist),percentile)
                    if central_tendency=="win_mean":
                        temp= mean(winsorize(np.array(dlist),cutting_ratio))
        upper.append(temp)
        temp=0
        dlist.clear()
    #print(upper)
    for k in range(1,np.shape(curr_arr)[0]):
        for i in range(np.shape(curr_arr)[0]):
            for j in range(np.shape(curr_arr)[1]):
                if(i==j+k):
                    dlist.append(curr_arr[i][j])
                    if central_tendency == "mean":
                        temp=sum(dlist)/len(dlist)
                    if central_tendency == "median":
                        temp= median(dlist)
                    if central_tendency == "trim_mean":
                        temp= stats.trim_mean(np.array(dlist),cutting_ratio)
                    if central_tendency == "percentile":
                        temp= np.percentile(np.array(dlist),percentile)
                    if central_tendency=="win_mean":
                        temp= mean(winsorize(np.array(dlist),cutting_ratio))
        lower.append(temp)
        temp=0
        dlist.clear()
    #print(lower)
    # new_arr=toeplitz(upper,lower).transpose()
    # #print(new_arr)
    # return(new_arr)
    mid_answer = np.copy(toeplitz(upper,lower).transpose())
    return mid_answer

def convert(arr,block_size=(3,4) , central_tendency = "win_mean",cutting_ratio =0.4,percentile =100,conversion_matrix="block_cir"):
    arr = np.array(arr)
    arr = arr.astype(float)
    #print(arr)
    row_begin =0
    row_end = block_size[0] - 1
    rows,cols = np.shape(arr)
    new_arr = np.copy(arr)
    print(rows)
    while ( row_end < rows ):
        print(row_end)
        col_begin = 0
        col_end = block_size[1] -1
        while ( col_end < cols) :
            curr_arr = np.copy(arr[row_begin:row_end+1,col_begin:col_end+1] )
            if(conversion_matrix=="block_cir" and block_size[0]==block_size[1]):
                mid_answer=tobc(curr_arr=curr_arr,block_size=block_size,central_tendency=central_tendency,cutting_ratio=cutting_ratio,percentile=percentile)
            elif(conversion_matrix=="Toeplitz"):
                mid_answer=toeplitz_faster(curr_arr=curr_arr,central_tendency=central_tendency,cutting_ratio =cutting_ratio,percentile =percentile)
            else:
                mid_answer=to_hankel(curr_arr=curr_arr,central_tendency=central_tendency,cutting_ratio =cutting_ratio,percentile =percentile)
                print("Entering hankel")


            new_arr[row_begin:row_end+1,col_begin:col_end+1]= mid_answer
            col_begin = col_begin + block_size[1]
            col_end = col_end + block_size[1]
        row_begin = row_begin + block_size[0]
        row_end = row_end + block_size[0]
    # print(new_arr)
    return new_arr

def split(arr,block_size=(3,3),central_tendency = "mean",cutting_ratio =0.4,percentile =100,conversion_matrix="block_cir"):
    arr = np.array(arr)
    arr = arr.astype(float)
    new_arr = np.copy(arr)
    # rows,cols = np.shape(arr)

    half_split = np.array_split(arr, 2,axis=1)
    # print(type(half_split[0]))
    # print(half_split[0])
    # print(half_split[1])
    half_split[0]=convert(half_split[0],block_size,central_tendency,cutting_ratio,percentile,conversion_matrix)
    # arr2=convert(half_split[1],block_size=(3,3),central_tendency="mean",conversion_matrix="toeplitz")
    # print(half_split[0])
    # print(half_split[1])
    new_arr=np.concatenate((half_split[0], half_split[1]), axis=1)
    # sprint(new_arr)


    return new_arr




if __name__ =="__main__":

    a = np.random.randint(3,5,[10,10])

    ans = convert(a,block_size=(3,4),central_tendency="mean",cutting_ratio=0.4,percentile=100,conversion_matrix="toeplitz")

