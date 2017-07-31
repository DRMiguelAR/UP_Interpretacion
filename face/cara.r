library(pixmap)
img<-read.pnm("face.ppm")
m<-img@red
mat1<-apply(m,2,rev)
image(t(mat1), axes=FALSE,col=grey(seq(0,1,length=256)))

