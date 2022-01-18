## FeatureCloud Flimma app

Description

![Workflow](../data/images/Mean.png)
 As input, it gets a CSV file, e.g., `data.csv`, from each client, calculates the local mean, and communicates it with the controller.
Depending on the received data, the controller, based on usage of the SMPC module, calculates the global mean and sends it back to clients.
Finally, clients write the global mean into an output file.

### Config

```angular2html
fc_mean:
  local_dataset:
    data: data.csv
  logic:
    mode: file
    dir: .
  axis: 0
  use_smpc: false
  result:
    mean: mean.txt
```
The mean app can be used with secure SMPC aggregation to aggregate local datasets in three possible ways based on `axis`.
`axis` can get one of these values:
- `None`: each client will send the mean value of its dataset(a scalar).
- `0`: the client will send a mean value for each column or feature.
- `1`: same as `0`, but the number of samples will also be sent to have a weighted average.
