library(data.table) # fread,install.packages("data.table") 
library(gplots) # heatmaps install.packages("gplots")
library(ggplot2) #install.packages("ggplot2")
library(ggthemes) # install.packages("ggthemes")
library(BiocManager) #install.packages("BiocManager")
library(edgeR) # BiocManager::install("edgeR")
library(ComplexHeatmap) # BiocManager::install("ComplexHeatmap") BiocManager::install("ComplexHeatmap")
library(magick) # raster install.packages("magick")
library(Rtools) #install.packages("Rtools") #a revoir
library(taRifx) #for remove.factors install.packages("taRifx") #a revoir
library(dendextend) # dendogram install.packages("dendextend")
library(circlize) # for colorRamp2 install.packages("circlize")
## survival curve
library(ggplot2)
library(survival)
library(survminer) ##install.packages("survminer")
library(remotes)
##
library(DescTools) # GTest install.packages("DescTools")
## Enrichment terms analysis
library(plotrix) #install.packages("plotrix")
library(multtest) # à revoir BiocManager::install("multtest")
##
library(Hmisc) #install.packages("Hmisc")
library(foreach) # %dopar% install.packages("foreach")
library(doMC) # for parallel computing install.packages("doMC", repos="http://R-Forge.R-project.org")
library(reshape2) #install.packages("reshape2") mel,decast

estrogenMainEffects <- read.csv("sample_gr.csv", sep=",", header=T,row.name=1)

# Create a graph adjacency based on correlation distances between genes in  pairwise fashion.
g <- graph.adjacency(
  as.matrix(as.dist(cor(t(estrogenMainEffects), method="pearson"))),
  mode="undirected",
  weighted=TRUE,
  diag=FALSE
)

# Simplfy the adjacency object
g <- simplify(g, remove.multiple=TRUE, remove.loops=TRUE)

# Colour negative correlation edges as blue
E(g)[which(E(g)$weight<0)]$color <- "darkblue"

# Colour positive correlation edges as red
E(g)[which(E(g)$weight>0)]$color <- "darkred"

# Convert edge weights to absolute values
E(g)$weight <- abs(E(g)$weight)

# Remove edges below absolute Pearson correlation 0.8
g <- delete_edges(g, E(g)[which(E(g)$weight<0.8)])

# Remove any vertices remaining that have no edges
g <- delete.vertices(g, degree(g)==0)

# Assign names to the graph vertices (optional)
V(g)$name <- V(g)$name

# Change shape of graph vertices
V(g)$shape <- "sphere"

# Change colour of graph vertices
V(g)$color <- "skyblue"

# Change colour of vertex frames
V(g)$vertex.frame.color <- "white"



# Multiply scaled vales by a factor of 10
scale01 <- function(x){(x-min(x))/(max(x)-min(x))}
vSizes <- (scale01(apply(estrogenMainEffects, 1, mean)) + 1.0) * 10

# Amplify or decrease the width of the edges
edgeweights <- E(g)$weight * 2.0

# Convert the graph adjacency object into a minimum spanning tree based on Prim's algorithm
mst <- mst(g, algorithm="prim")


# Plot the tree object
plot(
  mst,
  layout=layout.fruchterman.reingold,
  edge.curved=TRUE,
  vertex.size=vSizes,
  vertex.label.dist=-0.5,
  vertex.label.color="black",
  asp=FALSE,
  vertex.label.cex=0.6,
  edge.width=edgeweights,
  edge.arrow.mode=0,
  
  
  
  main="My first graph")


mst.communities <- edge.betweenness.community(mst, weights=NULL, directed=FALSE)
mst.clustering <- make_clusters(mst, membership=mst.communities$membership)
V(mst)$color <- mst.communities$membership + 1

par(mfrow=c(1,2))
plot(
  mst.clustering, mst,
  layout=layout.fruchterman.reingold,
  edge.curved=TRUE,
  vertex.size=vSizes,
  vertex.label.dist=-0.5,
  vertex.label.color="black",
  asp=FALSE,
  vertex.label.cex=0.6,
  edge.width=edgeweights,
  edge.arrow.mode=0,
  main="My first graph")

plot(
  mst,
  layout=layout.fruchterman.reingold,
  edge.curved=TRUE,
  vertex.size=vSizes,
  vertex.label.dist=-0.5,
  vertex.label.color="black",
  asp=FALSE,
  vertex.label.cex=0.6,
  edge.width=edgeweights,
  edge.arrow.mode=0,
  main="My  second graph ")

