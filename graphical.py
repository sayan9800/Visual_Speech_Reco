import numpy as np 
from matplotlib import pyplot as plt
import scipy as sp 

matrix_data = np.loadtxt('Thank_You.txt', delimiter = ',')
mask_mark = 0
sentence = 'Thank You'

matrix_data = matrix_data.reshape(-1, 13, 2)
matrix_data_x_fft = sp.fft.fftshift(sp.fft.fft(matrix_data[:,mask_mark,0]))
matrix_data_y_fft = sp.fft.fftshift(sp.fft.fft(matrix_data[:,mask_mark,1]))
t = np.linspace(0, 1, 25)
w = np.linspace(-np.pi, np.pi, len(matrix_data_x_fft))

for mask_mark in range(13):
	matrix_data_x_fft = sp.fft.fftshift(sp.fft.fft(matrix_data[:,mask_mark,0]))
	matrix_data_y_fft = sp.fft.fftshift(sp.fft.fft(matrix_data[:,mask_mark,1]))
	plt.subplot(2,1,1)
	plt.plot(t, matrix_data[:,mask_mark,0], 'g')
	plt.title('Lip coordinates of position ' + str(mask_mark+1) + ' - ' + '\"' + sentence + '\"')
	plt.grid(True)
	plt.ylabel('position - x')
	plt.subplot(2,1,2)
	plt.plot(t, matrix_data[:,mask_mark,1], 'b')
	plt.grid(True)
	plt.xlabel('time')
	plt.ylabel('position - y')
	plt.show()


	#print(np.abs(matrix_data_x_fft), np.abs(matrix_data_y_fft))

	# plt.subplot(2,1,1)
	# plt.plot(w, np.abs(matrix_data_x_fft), 'g')
	# plt.title('Fourier Transform Lip coordinates of position ' + str(mask_mark) + ' - ' + sentence)
	# plt.grid(True)
	# plt.ylabel('ft - position - x')
	# plt.subplot(2,1,2)
	# plt.plot(w, np.abs(matrix_data_y_fft), 'g')
	# plt.grid(True)
	# plt.xlabel('freq')
	# plt.ylabel('ft - position - y')
	# plt.show()