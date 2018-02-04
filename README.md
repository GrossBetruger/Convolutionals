### CAD CLASSIFIER - 3d CNN

<div align="center">
  <img src="http://vision.cs.princeton.edu/projects/2014/ModelNet/data/apple//apple_000000247/apple_000000247_thumb.jpg"><br><br>
</div>


## Installation

* Requires python 2.7.*

#### * install required packages using pip
```shell
$ pip install -r requirements.txt
```

### CNN classifier of 3d CAD models

* classifier_3d is a main module that can train, and test on CAD datasets 

* data is supplied as part of this repository in .MAT format (we are parsing it to numpy DS for you don't worry...), the data is based on Princeton's ModelNet40
 
* the module has two different cnn models (regular and concatenated convolutions)

* to train regular model run from command line

```shell
$ python classifier_3d.py train regular_network
```

* regular_network can be replaced with any other name (that will later be used to reference this model)

* to train concat model - do the same but with a name starting with 'concat'

```shell
$ python classifier_3d.py train concat_network
```

* to test your network run it in test mode (referring to it by the same name)

```shell
$ python classifier_3d.py test regular_network
```

### Data Augmentation 

* the module cad_data_set_generator.py also supports data augmentation:

```python
def prepare_data_set(dataset_dir, batch_size, channels, limit=None,
                     balanced=True, fuzzing_mode=False, num_of_voxels_to_augment=0):
```

* the 'num_of_voxels_to_augment' parameter sets the number of voxels to randomize in each CAD

* it defaults to 0, before training a network on augmented data it should be changed (to something in the order of a few hundreds)

* it's best to set it using the constant 'AUGMENTED_VOXELS' in classifier_3d.py

```python
AUGMENTED_VOXELS = 200
```


### Running a meta classifier on a few models 

* the module cad_meta_model.py is able to train and run a meta_model on a few cnn networks (which you must first train)

* by default it is wired to run on 3 trained networks named: model_conv3dregular10_v1, model_conv3dregular10_augmented_v1, model_conv3dconcat10_v1

* you should train 3 networks and then update the model paths within cad_meta_model.py

```python
raw_pred_reg = run_model(data, label, build_3dconv_cvnn, "model_conv3dregular10_v1", regular_counter)
raw_pred_aug = run_model(data, label, build_3dconv_cvnn, "model_conv3dregular10_augmented_v1", data_aug_counter)
raw_pred_concat = run_model(data, label, build_concat3dconv_cvnn, "model_conv3dconcat10_v1", concat_counter)
```

* the first two models must be trained with the regular cnn network (one should be trained with augmented data - see above) and the third must be trained as concatenated network (see above)

* after that - train the meta model as follows:

```shell
$ python cad_meta_model.py meta_model.data 
```

* where 'meta_model.data' is the name of the file that will contain the dataset to train the metamodel

* after running the meta model once you need to update the model name (example: 'meta_model.data') which is hard-coded in cad_meta_model.py (I know...) as follows: 

```python
machine_learning_data_set_path = "meta_model.data"
```

* then run it again, now you will also get results from the meta model (meta model will have 4 predictors - random forest, decision tree, logistic regression and svm)

* if you want to run the meta model on a different number of networks (or different kinds of networks) you should change the code in 'cad_meta_model.py' accordingly