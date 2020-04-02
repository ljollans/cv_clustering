#INPUTS: 𝑇 = Data
#PARAMS: 𝐵 = Monte Carlo simulations, 𝐻 = Inner resampling iterations, 𝐾 = maximum K,
#clustering algorithm = PAM (default)

### REFERENCE
#FOR 𝑏 = 1 … 𝐵:
    #Calculate random data 𝑄𝑏 (equations 1-3)
    #Calculate Euclidean distance matrix
    #FOR ℎ = 1 … 𝐻:
        #Resample 𝑄𝑏
        #Create or update the indicator matrix I (equation 4)
        #FOR 𝑘 = 1 … 𝐾:
            #Cluster distance matrix using clustering algorithm
            #Assign clustering to connectivity matrix 𝑀 (equation 5)
    #Normalise 𝑀 (equation 6)
    #Calculate CDF and reference PAC scores (equations 7-8)

### REAL DATA
#Calculate Euclidean distance matrix
#FOR ℎ = 1 … 𝐻:
    #Resample 𝑇
    #Create or update the indicator matrix I (equation 4)
    #FOR 𝑘 = 1 … 𝐾:
        #Cluster distance matrix using inner clustering algorithm
        #Assign clustering to connectivity matrix 𝑀 (equation 5)
#Normalise 𝑀 for 𝑘 = 1 … 𝐾 (equation 6)
#Calculate CDF then get real PAC score for each 𝑘 (equations 7-8)
### FINAL STEPS
#Calculate RCSI for 𝑘 = 1 … 𝐾 (equation 9)
#Calculate p values for 𝑘 = 1 … 𝐾 (equation 10)