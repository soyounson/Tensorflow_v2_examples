�	      �?      �?!      �?	�E]t�=@�E]t�=@!�E]t�=@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$      �?�I+��?A�Zd;�?Y=
ףp=�?*	     0�@2F
Iterator::Model��� �r�?!e�f#C@)Zd;�O��?1��9��oB@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapH�z�G�?!(<	�;@)�|?5^��?1�HFm�#4@:Preprocessing2�
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat+����?!,��l$�2@)��Q��?1n��w�0@:Preprocessing2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map!�rh���?!�����a@@)?5^�I�?1L�q֪A,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::ConcatenateV-��?!aq*?@)�v��/�?1�|�:�@:Preprocessing2�
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Range/�$��?!�%�8k� @)/�$��?1�%�8k� @:Preprocessing2U
Iterator::Model::ParallelMapV2y�&1��?!��K�q�?)y�&1��?1��K�q�?:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�� �rh�?!� ŀ'A�?)�~j�t��?1?�]�=�?:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch�I+��?!d���?)�I+��?1d���?:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zt?!�<�Z5�?){�G�zt?1�<�Z5�?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor����MbP?!�-}Ļ��?)����MbP?1�-}Ļ��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlice����MbP?!�-}Ļ��?)����MbP?1�-}Ļ��?:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensor����MbP?!�-}Ļ��?)����MbP?1�-}Ļ��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
host�Your program is HIGHLY input-bound because 29.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t34.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9�E]t�=@>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�I+��?�I+��?!�I+��?      ��!       "      ��!       *      ��!       2	�Zd;�?�Zd;�?!�Zd;�?:      ��!       B      ��!       J	=
ףp=�?=
ףp=�?!=
ףp=�?R      ��!       Z	=
ףp=�?=
ףp=�?!=
ףp=�?JCPU_ONLYY�E]t�=@b Y      Y@qHe}j�@"�
host�Your program is HIGHLY input-bound because 29.8% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nohigh"t34.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQ2"CPU: B 