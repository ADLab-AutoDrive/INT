# INT: Towards Infinite-frames 3D Detection with An Efficient Framework

It is natural to construct a multi-frame instead of a single-frame 3D detector for a continuous-time stream.
Although increasing the number of frames might improve performance, previous multi-frame studies only used
very limited frames to build their systems due to the dramatically increased computational and memory cost.
To address these issues, we propose a novel on-stream training and prediction framework that, in theory,
can employ an infinite number of frames while keeping the same amount of computation as a single-frame detector.
This infinite framework (INT), which can be used with most existing detectors, is utilized, for example,
on the popular CenterPoint, with significant latency reductions and performance improvements.
We've also conducted extensive experiments on two large-scale datasets, nuScenes and Waymo Open Dataset,
to demonstrate the scheme's effectiveness and efficiency.  By employing INT on CenterPoint,
we can get around 7% (Waymo) and 15% (nuScenes) performance boost with only 2~4ms latency overhead,
and currently SOTA on the Waymo 3D Detection leaderboard.

<p align="center"> <img src='resources/INT-Framework.png' align="center" height="230px"> </p>

[comment]: <> (![demo image]&#40;resources/INT-Framework.png&#41;)

INT is accepted by ECCV2022. Paper and code are on the way.

[comment]: <> (If you find this project useful, please cite:)

[comment]: <> (    @article{Xu2022INT,)

[comment]: <> (      title={INT: Towards Infinite-frames 3D Detection with An Efficient Framework},)

[comment]: <> (      author={Xu, Jianyun and Miao, Zhenwei and et al.},)

[comment]: <> (      journal={ECCV},)

[comment]: <> (      year={2022},)

[comment]: <> (    })



# Highlights

- **Simple and Fast:** INT is an on-stream multi-frame system made up of Memory Bank and Dynamic Training Sequence Length that can theoretically 
be trained and predicted using infinite frames while consuming similar computation and memory as a single-frame system.

- **SOTA**: Our 100-frames INT is currently SOTA on Waymo 3D Detection [leaderboard](https://waymo.com/open/challenges/2020/3d-detection/).

- **Extensible**: INT can be employed on most detectors, even for other tasks, like segmentation.

## Main results

#### 3D detection on Waymo val set

|         |  #Frame | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   |  latency(ms)  |
|---------|---------|--------|--------|---------|--------|-------|
|INT-1s   | 2       |  69.4     |  69.1      |  72.6       |   70.3     |   74.0    | 
|INT-1s   | 10      |  72.2    |  72.1      |  75.3       |   73.2     |  74.0     |
|INT-2s   | 2       |  70.8     |  68.7      |  73.1       |   70.8     |   78.9    | 
|INT-2s   | 10      |  73.3    |  71.9      |  75.6       |   73.6     |  78.9     |


#### 3D detection on Waymo test set

|         |  #Frame | Veh_L2 | Ped_L2 | Cyc_L2  | MAPH   |   latency(ms)  |
|---------|---------|--------|--------|---------|--------|-------|
|INT-2s | 10       |  76.2     |  72.8      |  72.7      |   73.9     |  78.9   |
|INT-2s | 100      |  77.6     |  74.0      |  74.1      |   75.2     |  78.9   | 



1s stands for 1-stage, 2s stands for 2-stage. 
All results are tested on a GeForce RTX 2070 SUPER with batch size 1.



[comment]: <> (## Use INT)

[comment]: <> (### Installation)

[comment]: <> (Please refer to [INSTALL]&#40;docs/INSTALL.md&#41; to set up libraries needed for distributed training and sparse convolution.)

[comment]: <> (### Benchmark Evaluation and Training)

[comment]: <> (Please refer to [GETTING_START]&#40;docs/GETTING_START.md&#41; to prepare the data. Then follow the instruction there to reproduce our detection and tracking results. All detection configurations are included in [configs]&#40;configs&#41;.)


[comment]: <> (### ToDo List)

[comment]: <> (- [ ] Support visualization with Open3D)

[comment]: <> (- [ ] Colab demo)

[comment]: <> (- [ ] Docker)



## License

INT is release under MIT license (see [LICENSE](LICENSE)). It is developed based on [CenterPoint](https://github.com/tianweiy/CenterPoint). 
Note that both nuScenes and Waymo datasets are under non-commercial licenses.





## Acknowlegement
We sincerely thank the following open-source code.

* [CenterPoint](https://github.com/tianweiy/CenterPoint)
