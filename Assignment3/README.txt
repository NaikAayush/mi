Data Preprocessing


The fields, Delivery Phase, Education and Residence had 4, 3 and 2 missing values respectively.
Further analysing them, the values are mostly repeated, thus we went with filling those missing values with the Mode of the particular column.

Assuming Haemoglobin content had some correlation with the Community, Age, and their Blood Pressure, we grouped their values by these fields respectively, and filled the missing values with the group mean. 
And to handle misisng values in the particular columns, decreased the features by one, to handle any missing data.
A similar approach was taken for the Blood Pressure column as well.

For Age, from analysing the data, we chose to take the mean of the community mean to fill in missing age values.
And the similar approach for the weight column as well.

In the end, as the result set values were skewed towards 1, we performed oversampling by duplicating the 0 values, to match the data spread.
