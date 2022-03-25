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


```angular2html
@article{Zolotareva2021,
 doi = {10.1186/s13059-021-02553-2},
 url = {https://doi.org/10.1186/s13059-021-02553-2},
 year = {2021},
 month = dec,
 publisher = {Springer Science and Business Media {LLC}},
 volume = {22},
 number = {1},
 author = {Olga Zolotareva and Reza Nasirigerdeh and Julian Matschinske and Reihaneh Torkzadehmahani and Mohammad Bakhtiari and Tobias Frisch and Julian Sp\"{a}th and David B. Blumenthal and Amir Abbasinejad and Paolo Tieri and Georgios Kaissis and Daniel R\"{u}ckert and Nina K. Wenke and Markus List and Jan Baumbach},
 title = {Flimma: a federated and privacy-aware tool for differential gene expression analysis},
 journal = {Genome Biology}
}
  
@misc{nasirigerdeh2021hyfed,
       title={HyFed: A Hybrid Federated Framework for Privacy-preserving Machine Learning},
       author={Reza Nasirigerdeh and Reihaneh Torkzadehmahani and Julian Matschinske and Jan Baumbach and Daniel Rueckert and Georgios Kaissis},
       year={2021},
       eprint={2105.10545},
       archivePrefix={arXiv},
       primaryClass={cs.LG}
}
```