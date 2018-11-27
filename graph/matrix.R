library("SomaticSignatures")
temp = read.csv(file.choose(), sep=",")
allData <- as.matrix(temp) 

m = nrow(allData)
n = ncol(allData)
k=5

wp <- rnmf(unlist(allData), k = k, showprogress = FALSE, my.seed = 100,alpha=0,beta=0,gamma=0)

xp = matrix(unlist(wp[1]),ncol=k)
yp = matrix(unlist(wp[2]),ncol=n)
wp = xp %*% yp
allData = matrix(allData,nrow=m,ncol=n)

n = norm(wp-allData)
print(n)