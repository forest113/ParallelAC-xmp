## Build

Use the following set of commands to clone and build:

```bash
$ git clone https://bitbucket.org/vgl_iisc/parallelac.git
$ cd parallelac
$ mkdir build
$ cd build
$ cmake ..
$ make
```

The commands above should generate two executables called **parallelac** and **parallelac-largeData**.

We have tested this code on a system running Ubuntu 18.10 with CUDA 10.1 having Nvidia GeForce GTX 1080 graphics card.

---

## Execute

The executables generated can be used to compute weighted alpha complexes for set of balls given in CRD format. The command to execute is:

```bash
$ parallelac <crd-file> <out-file> <sol-rad> <alpha>
```

On modern GPUs, computation of the alpha complex for input containing up to 100,000 balls should be possible using the command above. However, for larger inputs like huge protein complexes containing a few million atoms, the GPU memory may not be sufficient to process the whole complex in one go. In such cases, we break up the input into smaller chunks. Use the following command to process larger inputs:

```bash
$ parallelac-largeData <crd-file> <out-file> <sol-rad> <alpha> <chunk-size>
```

Where:

 1. `<crd-file>` is the input file in CRD format. A few sample CRD files are provided in the folder called **data**.
 2. `<out-file>` is the output alpha complex.
 3. `<sol-rad>` is the solvent radius which can be added to the atom radii before computation of the alpha complex.
 4. `<alpha>` is the alpha value.
 5. `<chunk-size>` a suggestion for the number of balls to be processed simulateneously on the GPU. A good suggestion for this parameter is 100000.

---

## Test Data

Some example CRD files are provided in the folder called **data**. It contains atomic representations of a few proteins obtained from [Protein Database](https://www.rcsb.org/). An example of a small protein is 1GRM which is provided as the file named `1grm.crd`. A typical medium sized protein contains a few thousand atoms, the file `1k4c.crd` is such an example. Lastly, `1x9p.crd` is an example of a fairly large protein complex containing more than 100,000 atoms. We would recommend using **parallelac-largeData** for such inputs.

---

## References

For more information about this algorithm, refer to follwoing papers. Please cite these publications if you use this method or the library in your work.


```
@inproceedings{masoodSocCG2020,
  author    = {Talha Bin Masood and
               Tathagata Ray and
               Vijay Natarajan},
  editor    = {Sergio Cabello and Danny Z. Chen},
  title     = {Parallel computation of alpha complex for biomolecules},
  booktitle = {36th International Symposium on Computational Geometry, SoCG 2020, June 23-26, 2020, Z\"{u}rich, Switzerland},
  series    = {LIPIcs},
  volume    = {164},
  pages     = {17:1--17:16},
  publisher = {Schloss Dagstuhl - Leibniz-Zentrum f{\"{u}}r Informatik},
  year      = {2020},
  doi       = {10.4230/LIPIcs.SoCG.2020.17},
  url       = {https://doi.org/10.4230/LIPIcs.SoCG.2020.17}
}

@article{masoodCGTA2020,
  author    = {Talha Bin Masood and
               Tathagata Ray and
               Vijay Natarajan},
  title     = {Parallel computation of alpha complexes for biomolecules},
  journal   = {Computational Geometry},
  volume    = {90},
  pages     = {101651},
  year      = {2020},
  issn      = {0925-7721},
  doi       = {https://doi.org/10.1016/j.comgeo.2020.101651},
  url       = {http://www.sciencedirect.com/science/article/pii/S0925772120300456}
}
```

---

## Acknowledgement

We would like to thank Prof. Sathish Vadhiyar and Nikhil Ranjanikar for helpful discussions and suggestions during the early phase of this work.

---

## Copyright

Copyright (c) 2019 Visualization & Graphics Lab (VGL), Indian Institute of Science. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
Author   : Talha Bin Masood

Contact  : talha [AT] iisc.ac.in