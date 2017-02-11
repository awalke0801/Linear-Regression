import numpy as np
import csv 
import math
# from sklearn.cluster import KMeans

#Reading the target variables 
f = csv.reader(open('Querylevelnorm_t.csv', 'rU'))
l = list(f)
target = np.array(l)
target = target.astype(np.float)
#Reading the sample dataset
f = csv.reader(open('Querylevelnorm_X.csv', 'rU'))
l = list(f)
fmat = np.matrix(l) 
fmat = fmat.astype(np.float)
#Dividing the sample into training testing and validation sets
train_mat = fmat[0:55000,:]
val_mat = fmat[55000:62323,:]
test_mat = fmat[62323:69623,:]
train_t = target[0:55000]
val_t = target[55000:62323]
test_t = target[62323:69623]


# #Kmeans tried to calculate the mean and variance using cluster formation 
# kmeans = KMeans(n_clusters = m)
# kmeans.fit(train_mat)

# centroids = kmeans.cluster_centers_
# variance = kmeans.inertia_ #Sum of distances of samples to their closest cluster center.

# # print centroids
# # print variance/k[0]
# variance = variance/k[0]
# inverse = 1/variance

print 'LeToR data Closed Form Solution'
#Values for hyperParameters
m = 2
lamb = 0
print 'M = ',m,' Lambda = ',lamb
#Calculating the variance matrix
k = train_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(train_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(train_mat[:,i]))
vari = np.diag(l)
#Calculating the MU matrix
bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(train_mat[cs-ss:cs,j]))
	bl.append(l)
mu_mat = np.matrix(bl)


#Calculating phi matrix
bl = []
inverse = np.linalg.inv(vari)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(train_mat[i,:] - mu_mat[j,:])*(inverse*(train_mat[i,:] - mu_mat[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_mat = np.matrix(bl)

#Calculating the weight vector 
I = np.identity(m)
w = (np.linalg.inv((lamb*(I) + (phi_mat.T)*(phi_mat))))*(phi_mat.T)*(train_t)
# print w 
#Calculating Error
ED = 0
for i in range(0,k[0]):
	ED = ED + math.pow((train_t[i] - ((w.T)*((phi_mat[i,:]).T))),2)
ED = 0.5*ED
EW = 0.5*((w.T)*(w))
E = ED + (lamb*(EW))
# print E

#Calculating Erms
Erms = math.sqrt(2*E/k[0])
print 'Training Erms: ',Erms


# print train_mat.shape, val_mat.shape, test_mat.shape, fmat.shape


# print '---------validation results---------'
k = val_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(val_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(val_mat[:,i]))
varv = np.diag(l)

bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(val_mat[cs-ss:cs,j]))
	bl.append(l)
mu_matv = np.matrix(bl)


bl = []
inverse = np.linalg.inv(varv)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(val_mat[i,:] - mu_matv[j,:])*(inverse*(val_mat[i,:] - mu_matv[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_matv = np.matrix(bl)


ED = 0
for i in range(0,k[0]):
	ED = ED + math.pow((val_t[i] - ((w.T)*((phi_matv[i,:]).T))),2)
ED = 0.5*ED
EW = 0.5*((w.T)*(w))
E = ED + (lamb*(EW))
# print E

#Calculating Erms
Erms = math.sqrt(2*E/k[0])
print 'Validation Erms: ',Erms





# print '---------test results---------'
k = test_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(test_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(test_mat[:,i]))
vart = np.diag(l)

bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(test_mat[cs-ss:cs,j]))
	bl.append(l)
mu_matt = np.matrix(bl)


bl = []
inverse = np.linalg.inv(vart)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(test_mat[i,:] - mu_matt[j,:])*(inverse*(test_mat[i,:] - mu_matt[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_matt = np.matrix(bl)


ED = 0
for i in range(0,k[0]):
	ED = ED + math.pow((test_t[i] - ((w.T)*((phi_matt[i,:]).T))),2)
ED = 0.5*ED
EW = 0.5*((w.T)*(w))
E = ED + (lamb*(EW))
# print E

#Calculating Erms
Erms = math.sqrt(2*E/k[0])
print 'Testing Erms: ',Erms


#----------------------------------------------------------------
print 'LeToR Data Stochastic gradient descent'

#Reading the target variables 
f = csv.reader(open('Querylevelnorm_t.csv', 'rU'))
l = list(f)
target = np.array(l)
target = target.astype(np.float)
#Reading the sample dataset
f = csv.reader(open('Querylevelnorm_X.csv', 'rU'))
l = list(f)
fmat = np.matrix(l) 
fmat = fmat.astype(np.float)
#Dividing the sample into training testing and validation sets
train_mat = fmat[0:55000,:]
val_mat = fmat[55000:62323,:]
test_mat = fmat[62323:69623,:]
train_t = target[0:55000]
val_t = target[55000:62323]
test_t = target[62323:69623]

m = 1
lamb = 0
eeta = 0.001
print 'M = ',m,' Lambda = ',lamb,' Eeta = ',eeta 

#Calculating the variance matrix
k = train_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(train_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(train_mat[:,i]))
vari = np.diag(l)
#Calculating the MU matrix
bl = []

ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(train_mat[cs-ss:cs,j]))
	bl.append(l)
mu_mat = np.matrix(bl)


#Calculating phi matrix
bl = []
inverse = np.linalg.inv(vari)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(train_mat[i,:] - mu_mat[j,:])*(inverse*(train_mat[i,:] - mu_mat[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_mat = np.matrix(bl)

w = (np.ones(m))
w = np.matrix(w) 
#print -(phi_mat[0,:].T)*(train_t[0] - (w*(phi_mat[0,:].T)))

w0 = w.T
# print w0 
# tk = []
# Emin = float('inf')
for i in range(0,k[0]):
	EW = w0
	ED = -(phi_mat[i,:].T)*(train_t[i] - ((w0.T)*(phi_mat[i,:].T)))
	E = ED + lamb*(EW)
	# if E.any()<Emin:
	# 	Emin = E.any() 
	w1 = -eeta*(E)
	# tk.append(np.amax(np.absolute(w1)))
	w0 = w0 + w1
# print Emin
# print E
# print w0 
# tk1 = np.array(tk)
# print np.amin(tk1)


EDse = 0
for i in range(0,k[0]):
	EDse = EDse + math.pow((train_t[i] - ((w0.T)*((phi_mat[i,:]).T))),2)
EDse = 0.5*EDse
EWse = 0.5*((w0.T)*(w0))
Ese = EDse + (lamb*(EWse))
# print Ese

Erms = math.sqrt(2*Ese/k[0])
print 'Training Erms: ',Erms




# print train_mat.shape, val_mat.shape, test_mat.shape, fmat.shape




# print '---------validation results---------'
k = val_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(val_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(val_mat[:,i]))
varv = np.diag(l)

bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(val_mat[cs-ss:cs,j]))
	bl.append(l)
mu_matv = np.matrix(bl)


bl = []
inverse = np.linalg.inv(varv)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(val_mat[i,:] - mu_matv[j,:])*(inverse*(val_mat[i,:] - mu_matv[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_matv = np.matrix(bl)


EDse = 0
for i in range(0,k[0]):
	EDse = EDse + math.pow((val_t[i] - ((w0.T)*((phi_matv[i,:]).T))),2)
EDse = 0.5*EDse
EWse = 0.5*((w0.T)*(w0))
Ese = EDse + (lamb*(EWse))
# print Ese

Erms = math.sqrt(2*Ese/k[0])
print 'Validation Erms: ',Erms





# print '---------test results---------'
k = test_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(test_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(test_mat[:,i]))
vart = np.diag(l)

bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(test_mat[cs-ss:cs,j]))
	bl.append(l)
mu_matt = np.matrix(bl)


bl = []
inverse = np.linalg.inv(vart)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(test_mat[i,:] - mu_matt[j,:])*(inverse*(test_mat[i,:] - mu_matt[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_matt = np.matrix(bl)


EDse = 0
for i in range(0,k[0]):
	EDse = EDse + math.pow((test_t[i] - ((w0.T)*((phi_matt[i,:]).T))),2)
EDse = 0.5*EDse
EWse = 0.5*((w0.T)*(w0))
Ese = EDse + (lamb*(EWse))
# print Ese

Erms = math.sqrt(2*Ese/k[0])
print 'Testing Erms: ',Erms





# ----------Synthetic Data Closed Form Solution ----------
print 'Synthetic Data Closed Form Solution'

#Reading the target variables 
f = csv.reader(open('output.csv', 'rU'))
l = list(f)
target = np.array(l)
target = target.astype(np.float)
#Reading the sample dataset
f = csv.reader(open('input.csv', 'rU'))
l = list(f)
fmat = np.matrix(l) 
fmat = fmat.astype(np.float)
#Dividing the sample into training testing and validation sets
train_mat = fmat[0:16000,:]
val_mat = fmat[16000:18000,:]
test_mat = fmat[18000:20000,:]
train_t = target[0:16000]
val_t = target[16000:18000]
test_t = target[18000:20000]

#Assigning parameter values 
m = 4
lamb = 1
print 'm = ',m,' Lambda = ',lamb
#Calculating the variance matrix
k = train_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(train_mat[:,i]) == 0:
	 	l.append(0.00001)
	 	# l.append(1)
	else:
		l.append(np.var(train_mat[:,i]))
		# l.append(1)
vari = np.diag(l)
# print vari.shape
#Calculating the MU matrix
bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(train_mat[cs-ss:cs,j]))
	bl.append(l)
mu_mat = np.matrix(bl)


#Calculating phi matrix
bl = []
inverse = np.linalg.inv(vari)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(train_mat[i,:] - mu_mat[j,:])*(inverse*(train_mat[i,:] - mu_mat[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_mat = np.matrix(bl)

#Calculating the weight vector 
I = np.identity(m)
w = (np.linalg.inv((lamb*(I) + (phi_mat.T)*(phi_mat))))*(phi_mat.T)*(train_t)
# print w 
#Calculating Error
ED = 0
for i in range(0,k[0]):
	ED = ED + math.pow((train_t[i] - ((w.T)*((phi_mat[i,:]).T))),2)
ED = 0.5*ED
EW = 0.5*((w.T)*(w))
E = ED + (lamb*(EW))
# print E

#Calculating Erms
Erms = math.sqrt(2*E/k[0])
print 'Training Erms: ',Erms


# print train_mat.shape, val_mat.shape, test_mat.shape, fmat.shape

# print '---------validation results---------'
k = val_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(val_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(val_mat[:,i]))
varv = np.diag(l)

bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(val_mat[cs-ss:cs,j]))
	bl.append(l)
mu_matv = np.matrix(bl)


bl = []
inverse = np.linalg.inv(varv)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(val_mat[i,:] - mu_matv[j,:])*(inverse*(val_mat[i,:] - mu_matv[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_matv = np.matrix(bl)


ED = 0
for i in range(0,k[0]):
	ED = ED + math.pow((val_t[i] - ((w.T)*((phi_matv[i,:]).T))),2)
ED = 0.5*ED
EW = 0.5*((w.T)*(w))
E = ED + (lamb*(EW))
# print E

#Calculating Erms
Erms = math.sqrt(2*E/k[0])
print 'Validation Erms: ',Erms





# print '---------test results---------'
k = test_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(test_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(test_mat[:,i]))
vart = np.diag(l)

bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(test_mat[cs-ss:cs,j]))
	bl.append(l)
mu_matt = np.matrix(bl)


bl = []
inverse = np.linalg.inv(vart)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(test_mat[i,:] - mu_matt[j,:])*(inverse*(test_mat[i,:] - mu_matt[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_matt = np.matrix(bl)


ED = 0
for i in range(0,k[0]):
	ED = ED + math.pow((test_t[i] - ((w.T)*((phi_matt[i,:]).T))),2)
ED = 0.5*ED
EW = 0.5*((w.T)*(w))
E = ED + (lamb*(EW))
# print E

#Calculating Erms
Erms = math.sqrt(2*E/k[0])
print 'Testing Erms: ',Erms



print 'Synthetic Data Stochastic gradient descent'

#Reading the target variables 
f = csv.reader(open('output.csv', 'rU'))
l = list(f)
target = np.array(l)
target = target.astype(np.float)
#Reading the sample dataset
f = csv.reader(open('input.csv', 'rU'))
l = list(f)
fmat = np.matrix(l) 
fmat = fmat.astype(np.float)
#Dividing the sample into training testing and validation sets
train_mat = fmat[0:16000,:]
val_mat = fmat[16000:18000,:]
test_mat = fmat[18000:20000,:]
train_t = target[0:16000]
val_t = target[16000:18000]
test_t = target[18000:20000]

m = 10
lamb = 0
eeta = 0.0001

print 'M = ',m,' Lambda = ',lamb,' Eeta = ',eeta
#Calculating the variance matrix
k = train_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(train_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(train_mat[:,i]))
vari = np.diag(l)
#Calculating the MU matrix
bl = []

ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(train_mat[cs-ss:cs,j]))
	bl.append(l)
mu_mat = np.matrix(bl)


#Calculating phi matrix
bl = []
inverse = np.linalg.inv(vari)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(train_mat[i,:] - mu_mat[j,:])*(inverse*(train_mat[i,:] - mu_mat[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_mat = np.matrix(bl)
# print phi_mat.shape

w = (np.ones(m))
w = np.matrix(w) 
#print -(phi_mat[0,:].T)*(train_t[0] - (w*(phi_mat[0,:].T)))

w0 = w.T
# print w0 
# tk = []
# Emin = float('inf')
for i in range(0,k[0]):
	EW = w0
	ED = -(phi_mat[i,:].T)*(train_t[i] - ((w0.T)*(phi_mat[i,:].T)))
	E = ED + lamb*(EW)
	# print np.amax(E)
	# if np.absolute(np.amax(E)) < 0.03:
	# 	break
	# if np.absolute(np.amax(E))<Emin:
	#  	Emin = np.amax(E)
	#  	wmin = -eeta*(E) 
	w1 = -eeta*(E)
	# tk.append(np.amax(np.absolute(w1)))
	# if np.amax(np.absolute(w1))<0.0000001:
	# 	break
	w0 = w0 + w1
# print Emin
# print E
# tk1 = np.array(tk)
# print w0 
# print wmin 
# print np.amin(tk1)

EDse = 0
for i in range(0,k[0]):
	EDse = EDse + math.pow((train_t[i] - ((w0.T)*((phi_mat[i,:]).T))),2)
EDse = 0.5*EDse
EWse = 0.5*((w0.T)*(w0))
Ese = EDse + (lamb*(EWse))
# print Ese

Erms = math.sqrt(2*Ese/k[0])
print 'Training Erms: ',Erms




# print train_mat.shape, val_mat.shape, test_mat.shape, fmat.shape




# print '---------validation results---------'
k = val_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(val_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(val_mat[:,i]))
varv = np.diag(l)

bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(val_mat[cs-ss:cs,j]))
	bl.append(l)
mu_matv = np.matrix(bl)


bl = []
inverse = np.linalg.inv(varv)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(val_mat[i,:] - mu_matv[j,:])*(inverse*(val_mat[i,:] - mu_matv[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_matv = np.matrix(bl)


EDse = 0
for i in range(0,k[0]):
	EDse = EDse + math.pow((val_t[i] - ((w0.T)*((phi_matv[i,:]).T))),2)
EDse = 0.5*EDse
EWse = 0.5*((w0.T)*(w0))
Ese = EDse + (lamb*(EWse))
# print Ese

Erms = math.sqrt(2*Ese/k[0])
print 'Validation Erms: ',Erms





# print '---------test results---------'
k = test_mat.shape
l = []
for i in range(0,k[1]):
	if np.var(test_mat[:,i]) == 0:
	 	l.append(0.00001)
	else:
		l.append(np.var(test_mat[:,i]))
vart = np.diag(l)

bl = []
ss = k[0]/m
cs = 0
for i in range(0,m):
	cs = cs + ss
	l = []
	for j in range(0,k[1]):
		l.append(np.mean(test_mat[cs-ss:cs,j]))
	bl.append(l)
mu_matt = np.matrix(bl)


bl = []
inverse = np.linalg.inv(vart)
for i in range(0,k[0]):
	l = []
	for j in range(0,m):
		d = math.expm1(-0.5*(test_mat[i,:] - mu_matt[j,:])*(inverse*(test_mat[i,:] - mu_matt[j,:]).T))
		#d = float((train_mat[i,:] - mu_mat[j,:])*(inverse*((train_mat[i,:] - mu_mat[j,:]).T)))
		l.append(d)
	bl.append(l)
phi_matt = np.matrix(bl)


EDse = 0
for i in range(0,k[0]):
	EDse = EDse + math.pow((test_t[i] - ((w0.T)*((phi_matt[i,:]).T))),2)
EDse = 0.5*EDse
EWse = 0.5*((w0.T)*(w0))
Ese = EDse + (lamb*(EWse))
# print Ese

Erms = math.sqrt(2*Ese/k[0])
print 'Testing Erms: ',Erms

