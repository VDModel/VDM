# VDM


Vegetation Dynamic Model


The simulation of vegetation dynamics is essential for guiding regional ecological remediation and environmental management. Recent progress in deep learning methods has provided possible solutions to vegetation simulations. The gated recurrent unit (GRU) is one of the latest deep learning algorithms that can effectively process dynamic data. However, static and dynamic data, which typically coexist in the datasets of vegetation dynamic changes, are typically processed indistinguishably. To efficiently extract spatiotemporal patterns and improve our ability to simulate potential vegetation changes, we introduced GRU into vegetation simulation and further amended the original structure of GRU according to the characteristics of simulation dataset. The new model, the vegetation dynamics model (VDM), can independently process static and dynamic data using a more appropriate algorithm, thereby improving the accuracy of simulation. Moreover, we presented a model test applied in the Luntai Desert-Oasis Ecotone in Northwest China and compared the performance of the VDM with baseline models. The results showed that the VDM produced a 7.51% higher coefficient of determination (R2) value, 7.51% higher adjusted R2 value, 16.67% lower mean squared error, and 10.78% lower mean absolute error than those of the GRU, which is the best baseline model. The proposed VDM is the first GRU-based simulation model of vegetation dynamics that has the potential to detect the time-order characteristics of dynamic factors by comprehensively considering the static information that affects vegetation changes. Moreover, the flexibility of the VDM, in combination with the wide availability of data from different data sources, aids the broader application of the VDM. 


Keywords: vegetation dynamic, gated recurrent unit neural network, static data, dynamic data, simulation model 
