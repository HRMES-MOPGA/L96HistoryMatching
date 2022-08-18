scale <- function(df){
    return(lapply(1:ncol(df), function(i) {c(min(df[,i]), max(df[,i]))}))
}
rangeScale <- function(df, scaler, range = c(-1, 1)){
    for (i in 1:ncol(df)){
        min <- scaler[[i]][1]
        max <- scaler[[i]][2]
        df[,i] = (range[2] - range[1])*(df[,i] - min)/(max - min) + range[1]
    }
    return(df)
}
                                        
rangeUnscale <- function(df, scaler, range = c(-1, 1)){
    for (i in 1:ncol(df)){
        min <- scaler[[i]][1]
        max <- scaler[[i]][2]
        df[,i] = (max - min)*(df[,i] - range[1])/(range[2] - range[1]) + min
    }
    return(df)
}

#################### Unpreprocess caret ###################################
#https://github.com/jknowles/ModelEWS/blob/master/man/unPreProc.Rd

unPreProc <- function(preProc, data){
  stopifnot(class(preProc) == "preProcess")
  stopifnot(class(data) == "data.frame")
  for(i in names(preProc$mean)){
    tmp <- data[, i] * preProc$std[[i]] + preProc$mean[[i]]
    data[, i] <- tmp
  }
  return(data)  
}

################### KMEANS ################################################
#https://bgstieber.github.io/post/an-introduction-to-the-kmeans-algorithm/

kmeansAIC <- function(fit){

  m = ncol(fit$centers) 
  k = nrow(fit$centers)
  D = fit$tot.withinss
  return(D + 2*m*k)
  
}

kmeansBIC <- function(fit){
  m = ncol(fit$centers) 
  n = length(fit$cluster)
  k = nrow(fit$centers)
  D = fit$tot.withinss
  return(D + log(n) * m * k) # using log(n) instead of 2, penalize model complexity
}

kmeans2 <- function(data, center_range, iter.max, nstart, plot = TRUE){
  
  #fit kmeans for each center
  all_kmeans <- lapply(center_range, 
                       FUN = function(k) 
                         kmeans(data, center = k, iter.max = iter.max, nstart = nstart))
  
  #extract AIC from each
  all_aic <- sapply(all_kmeans, kmeansAIC)
  #extract BIC from each
  all_bic <- sapply(all_kmeans, kmeansBIC)
  #extract tot.withinss
  all_wss <- sapply(all_kmeans, FUN = function(fit) fit$tot.withinss)
  #extract between ss
  btwn_ss <- sapply(all_kmeans, FUN = function(fit) fit$betweenss)
  #extract totall sum of squares
  tot_ss <- all_kmeans[[1]]$totss
  #put in data.frame
  clust_res <- 
    data.frame('Clusters' = center_range, 
             'AIC' = all_aic, 
             'BIC' = all_bic, 
             'WSS' = all_wss,
             'BSS' = btwn_ss,
             'TSS' = tot_ss)
  #plot or no plot?
  if(plot){
    par(mfrow = c(2,2))
    with(clust_res,{
      plot(Clusters, AIC)
      plot(Clusters, BIC)
      plot(Clusters, WSS, ylab = 'Within Cluster SSE')
      plot(Clusters, BSS / TSS, ylab = 'Prop of Var. Explained')
    })
  }else{
    return(clust_res)
  }
  
}
