#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 17:36:58 2024

@author: alperen
"""

# importing
import numpy as np

# numpy basics

array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])  # 1*15 vector

print(array.shape)

a = array.reshape(3, 5) # 3*5 vector yaptık.
print("shape: ", array.shape)
print("dimension: ", a.ndim)

print("Data type: ", a.dtype.name)

print("size: ", a.size)

print("type: ", type(a))

array1 = np.array([[1,2,3,4], [5,6,7,8], [9,8,7,5]]) # 3*4 vector

zeros = np.zeros((3,4)) #Bize 0 lardan oluşan 3'e 4 bir matriks(çok boyutlu dizi) verir.

zeros[0,0] = 5
print(zeros)

ones_np = np.ones((3,4))
print(ones_np)

empty_np = np.empty((2,3))
print(empty_np) # 0 dan büyük çok çok küçük satılarla bir array oluşturur.

numpy_arange_list = np.arange(10,50,5) 
print(numpy_arange_list)

numpy_linspace_list = np.linspace(10,50,11) #Her adımda aynı birim ile artar.
print(numpy_linspace_list)

# %% 

np_array1 = np.array([1,2,3])
np_array2 = np.array([4,5,6])
print(np_array1+np_array2)
print(np_array1-np_array2)
print(np_array1**2)

print(np.sin(np_array1))

print(np_array1<2)

np_array3 = np.array([[1,2,3], [4,5,6]])
np_array4 = np.array([[1,2,3], [4,5,6]])

# element wise prodcut
print(np_array3*np_array4)

# matrix prodcut
np_array3.dot(np_array4.T) # matris çarpımı için np_array4'ün Transpozunu aldık.

np_random_array = np.random.random((5,5))
print(np_random_array)

print(np_random_array.sum())
print(np_random_array.max())
print(np_random_array.min())

print(np_random_array.sum(axis=0)) #sadece sütunları toplar.
print(np_random_array.sum(axis=1)) #sadece satırları toplar.

print(np.sqrt(np_random_array))
print(np.square(np_random_array))

print(np.add(np_random_array, np_random_array))

# %% indexing and slicing
import numpy as np
array = np.array([1,2,3,4,5,6,7])   #  vector dimension = 1

print(array[0])

print(array[0:4])

reverse_array = array[::-1]
print(reverse_array)

array1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])

print(array1[1,1])   # 1. satır 1.sütun 

print(array1[:,1]) # Tüm satırların 1. sütünu

print(array1[1,1:4]) # ilk satırın 1,2,3. değerleri.

print(array1[-1,:]) # Son satırın tamamı.
print(array1[:,-1]) # Tüm satırların son değerleri.

# %%
# shape manipulation
array = np.array([[1,2,3],[4,5,6],[7,8,9]])

# flatten
flat_array = array.ravel() # Arrayi tek boyutlu hale getirir.

old_array = flat_array.reshape(3,3) # Tekrar 3'e 3 yaptık.
print(old_array)

old_array_transpose =  old_array.T
print(old_array_transpose)

# Önemli Not --> resize = Şeklini değiştirdiği arrayin(matrisin) değerini 
# kalıcı değiştirir. reshape = Ayrı bir değişkene eşitlenmediği sürece şeklini
# değiştirdiği arrayin değişimini hafızaya kaydetmez.

# %% stacking arrays

array1 = np.array([[1,2],[3,4]])
array2 = np.array([[-1,-2],[-3,-4]])

# veritical
#array([[1, 2],
#       [3, 4]])
#array([[-1, -2],
#       [-3, -4]])
array3 = np.vstack((array1,array2))

# horizontal
#array([[1, 2],[-1, -2],
#       [3, 4]],[-3, -4]]

array4 = np.hstack((array1,array2))

print(array3)
print(array4)

#%% convert and copy
liste = [1,2,3,4]   # list

array = np.array(liste) #np.array

liste2 = list(array)

a = np.array([1,2,3])

b = a
b[0] = 5
c = a
print(a)
print(b)
print(c)
# Sadece b değişse de memoryde ortak kaydedilmeleri sebebiyle hepsi değişti.

d =  np.array([1,2,3])

e = d.copy()
e[0] = 5
f = d.copy()

print(d)
print(e)
print(f)

# copy methodu ile sadece e nin değeri değişti.


























