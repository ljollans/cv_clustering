library(M3C)
library(NMF) # loading for aheatmap plotting function
library(gplots) # loading this for nice colour scale
library(ggsci) # more cool colours

# now we have loaded the mydata and desx objects (with the package automatically)
# mydata is the expression data for GBM
# desx is the annotation for this data

## ---- message=FALSE, results='hide'----------------------------------------
PCA1 <- pca(mydata)

## ---- message=FALSE, results='hide'----------------------------------------
res <- M3C(mydata, cores=1, seed = 123, des = desx, removeplots = TRUE)

## --------------------------------------------------------------------------
res$scores

## ----fig.show = 'hide'-----------------------------------------------------
# get the data out of the results list (by using $ - dollar sign)
data <- res$realdataresults[[4]]$ordered_data # this is the data
annon <- res$realdataresults[[4]]$ordered_annotation # this is the annotation
ccmatrix <- res$realdataresults[[4]]$consensus_matrix # this is the consensus matrix

# normalise and scale the data
data <- t(scale(t(data))) # z-score normalise each row (feature)
data <- apply(data, 2, function(x) ifelse(x > 4, 4, x)) # compress data within range
data <- apply(data, 2, function(x) ifelse(x < -4, -4, x)) # compress data within range

# get some cool colour palettes from the ggsci package and RColourBrewer
ann_colors <- ggsci::pal_startrek("uniform")(4) # star trek palette
ann_colors2 <- ggsci::pal_futurama()(4) # futurama palette
pal <- rev(colorRampPalette(RColorBrewer::brewer.pal(10, "RdBu"))(256))
NMF::aheatmap(data, annCol = annon, scale = 'row', Colv = NA, distfun = 'pearson',
              color = gplots::bluered(256), annColors = list(class=ann_colors, consensuscluster=ann_colors2))


## ----fig.show = 'hide'-----------------------------------------------------
# time to plot the consensus matrix for the optimal cluster decision
ccmatrix <- res$realdataresults[[4]]$consensus_matrix # pull out the consensus matrix from the k = 4 object
pal <- rev(colorRampPalette(RColorBrewer::brewer.pal(9, "Reds"))(256)) # get some nice colours
NMF::aheatmap(ccmatrix, annCol = annon, Colv = NA, Rowv = NA,
              color = rev(pal), scale = 'none') # plot the heatmap

## ----fig.show = 'hide',message=FALSE,results='hide'------------------------
PCA2 <- pca(res, K=4)

## ----fig.show = 'hide',message=FALSE,results='hide'------------------------
res <- clustersim(225, 900, 8, 4, 0.75, 0.025, print = FALSE, seed=123)
